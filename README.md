# NextPOI — LLM-only Next POI Prediction

本仓库当前只保留 `NYC + llm-only` 主线，用于下一个兴趣点预测（Next POI Prediction）。

推荐把系统理解为一个三阶段的可解释流水线：

```text
build_profiles.py  ->  predict.py  ->  evaluate.py
 user grounding        llm-only        metrics
```

核心设计：

1. 用 `train + valid` 轨迹为每个用户构建叙事性出行画像。
2. 用用户画像嵌入和空间访问重叠计算相似用户。
3. 对每个测试轨迹，先构建候选池，再做时间感知的 LLM 意图推断与候选缩减，最后由 LLM 输出 top-10 排名。
4. 用 `Hit@K / N@K / MRR` 评估整体与分层表现。

当前仓库不再包含：

- reranker
- hybrid inference
- SFT / 微调主线

---

## 数据集目录

```
datasets/
├── nyc/
│   ├── trips_train.csv
│   ├── trips_valid.csv
│   ├── trips_test.csv
│   ├── loc2id
│   ├── user_index.json
│   ├── prompts/
│   └── prompts_refined/
├── ca/
│   ├── ca_train.jsonl
│   └── ca_test.jsonl
└── tky/
    ├── tky_train.jsonl
    └── tky_test.jsonl
```

默认主线数据路径是 `datasets/nyc`。当前代码直接适配的是 NYC 这一套项目格式数据；`datasets/ca` 和 `datasets/tky` 目前保留为 CoMaPOI 原始 JSONL 数据。

轨迹数据格式：每条记录包含 `user_id`、`traj_id`，以及由 `"lon,lat,timestamp"` 字符串组成的 checkin 列表。测试集预测任务定义为：给定一条轨迹的前 N-1 个 checkin，预测第 N 个位置。

---

## 环境配置

```bash
pip install pandas scipy geohash2 tqdm geohash2
pip install torch transformers accelerate bitsandbytes
pip install huggingface_hub FlagEmbedding openai
```

**配置 API**：
```bash
# 画像嵌入模型
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3

# 项目根目录 .env（推荐）
cat > .env <<'EOF'
OPENAI_API_KEY=your_key_here
# OPENAI_BASE_URL=https://your-proxy-or-compatible-endpoint/v1
EOF
```

LLM 请求通过 OpenAI API 调用；代码会优先读取项目根目录下的 `.env`，也兼容直接 `export` 环境变量。本地只使用 `bge-m3` 作为嵌入模型。

---

## 推荐主线配置

目前仓库内已有实验记录里，综合表现最好的主线设置是：

- 数据集：`datasets/nyc`
- 方法：`llm-only`
- 候选保留：`forced_include_n = 3`
- 用户画像模型：`gpt-5.4`
- 意图推断 / 候选预过滤模型：`gpt-5.4`
- 最终排序模型：`gpt-5.4-mini`

对应存档结果可参考：

- `results/snapshots/20260414_ablation_forced3/evaluation_results.json`
- `results/ablation_forced_include_n_20260414.md`

---

## 使用流程

### Step 1 — 构建用户画像与相似度矩阵

```bash
python build_profiles.py
```

- 调用 **`gpt-5.4`** 为每位用户生成 ~200 词的叙事性出行画像，缓存至 `cache/profiles/{user_id}.json`
- 使用 **bge-m3** 对画像做文本嵌入，结合地理区域（geohash）Jaccard 相似度，构建 1,055×1,055 用户相似度矩阵，缓存至 `cache/similarity.pkl`
- **支持断点续传**：已存在的画像缓存文件会自动跳过，中断后重新运行不会重复调用 API
- **支持多线程**：可通过 `--workers` 并发发起 LLM 请求，加快画像构建

可选参数：
```bash
python build_profiles.py --user-id 1         # 仅构建单个用户的画像（调试用）
python build_profiles.py --recompute         # 忽略缓存，全量重建
python build_profiles.py --workers 16        # 并行构建画像（注意 API rate limit）
python build_profiles.py --skip-similarity   # 只构建画像，跳过相似度计算
```

### Step 2 — 预测下一个 POI

```bash
python predict.py
```

对 test 集全部 2,698 条轨迹逐一预测：

1. 读取用户画像和相似度矩阵
2. 构建原始候选池（~100 个 POI）作为 recall 召回层
3. 通过时间感知的 LLM 预过滤与最终 LLM 排序生成 top-10 结果
4. 结果缓存至 `cache/predictions/{traj_id}.json`，**支持断点续传**

可选参数：
```bash
python predict.py --traj-id test_0      # 仅预测单条轨迹（调试用）
python predict.py --dry-run             # 打印 LLM prompt，不调用 API
python predict.py --workers 4           # 并行调用（注意 API rate limit）
python predict.py --forced-include-n 3  # nearby 候选强制保留数量；推荐默认值
python predict.py --recompute           # 忽略缓存，重新预测所有轨迹
```

推荐主线命令：

```bash
python build_profiles.py --build-transitions
python build_profiles.py --workers 16
python predict.py --workers 4 --forced-include-n 3
python evaluate.py
```

如果后续需要切换到别的数据目录，可以在运行前设置：

```bash
export NEXTPOI_DATA_DIR=/path/to/dataset_dir
```

### Step 3 — 评估

```bash
python evaluate.py
```

读取 `cache/predictions/` 中的所有预测结果，与 ground truth 对比，计算并打印：

- **整体指标**：Hit@1、Hit@5、Hit@10、N@1、N@5、N@10、MRR、recall_in_top10
- **分层指标**：按用户数据丰富程度、测试轨迹长度、目标时段分别统计

结果保存至 `results/evaluation_results.json`。

```bash
# 指定自定义预测目录
python evaluate.py --predictions-dir /path/to/predictions/
```

---

## 项目结构

```
NextPOI/
├── build_profiles.py      # 入口：Stage 1+2（画像构建 + 相似度）
├── predict.py             # 入口：Stage 3（llm-only 预测）
├── evaluate.py            # 入口：Stage 4（指标评估）
│
├── src/
│   ├── config.py          # 路径、模型名、超参数（统一配置）
│   ├── local_llm.py       # LLM 推理封装
│   ├── data_loader.py     # 数据加载、坐标解析、KD-Tree 空间查询
│   ├── profile_builder.py # Stage 1：gpt-5.4 生成用户画像
│   ├── user_similarity.py # Stage 2：geohash Jaccard + bge-m3 余弦相似度
│   ├── candidate_selector.py  # Stage 3a：原始候选池构建
│   ├── llm_prefilter.py   # Stage 3b：意图推断 + LLM 预过滤
│   ├── llm_agent.py       # Stage 3c：gpt-5.4-mini 预测 agent
│   ├── evaluator.py       # Stage 4：Acc@K / MRR 计算
│   └── utils.py           # haversine、日志等工具
│
├── models/                # 本地模型权重（huggingface-cli 下载）
│   └── bge-m3/
│
├── cache/
│   ├── profiles/          # {user_id}.json — 用户画像缓存
│   ├── similarity.pkl     # 用户相似度矩阵
│   ├── intent/            # {traj_id}.json — 意图推断缓存
│   ├── prefilter/         # {traj_id}.json — 预过滤结果缓存
│   └── predictions/       # {traj_id}.json — 最终预测结果缓存
│
└── results/
    └── evaluation_results.json
```

---

## LLM 配置

LLM 现在通过 `src/local_llm.py` 调用 OpenAI API；支持从项目根目录 `.env` 或 shell 环境变量读取配置。

| 用途 | 模型 | 调用次数 |
|------|------|---------|
| 用户画像构建 | `gpt-5.4` | ~1,055 次（一次性离线） |
| 出行意图推断 | `gpt-5.4` | ~2,698 次 |
| 候选预过滤 | `gpt-5.4` | ~2,698 次 |
| 下一位置预测 | `gpt-5.4-mini` | ~2,698 次 |
| 画像文本嵌入 | `BAAI/bge-m3` | ~1,055 次（一次性离线） |

模型路径可在 `src/config.py` 中修改：

```python
PROFILE_LLM_MODEL    = "gpt-5.4"
PREDICTION_LLM_MODEL = "gpt-5.4-mini"
INTENT_LLM_MODEL     = "gpt-5.4"
PREFILTER_LLM_MODEL  = "gpt-5.4"
EMBEDDING_MODEL      = "BAAI/bge-m3"
```

---

## 方法定位

如果你是按论文主线使用这个仓库，建议只围绕下面这条线展开：

- 数据集：`datasets/nyc`
- 方法：`llm-only`
- 关键贡献：用户 grounding、时间感知 shortlist、最终 LLM 排序

不建议把下面这些作为当前版本的主线：

- reranker
- hybrid
- 微调版本
- 其它数据格式转换实验

---

## 候选召回率分析

由于测试轨迹跨越多天、空间跨度大，候选筛选策略直接影响系统上限：

| 策略 | Recall@N |
|------|----------|
| 纯空间（KD-Tree top-100） | ~37% |
| 纯空间（KD-Tree top-500） | ~52% |
| **历史优先（当前策略，top-100）** | **~51.5%** |

当前策略中 61% 的 ground truth 出现在用户自身的训练历史中，因此优先分配历史访问槽位。Recall@100 ≈ 51.5% 是本系统预测精度的理论上限。
