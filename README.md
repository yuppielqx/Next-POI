# NextPOI — 基于 LLM Agent 的下一个兴趣点预测

基于 foursquare_NYC 数据集，使用 GPT-5 系列模型进行下一个兴趣点（Next POI）预测。

## 系统概述

系统分为三个独立阶段，依次执行：

```
build_profiles.py  →  predict.py  →  evaluate.py
   (画像 + 相似度)      (LLM 预测)      (指标计算)
```

**核心思路：**
1. 用 train + val 轨迹为每个用户构建出行画像（`gpt-5.4` 生成叙事性描述）
2. 基于画像嵌入（bge-m3）+ 地理区域重叠（geohash Jaccard）计算用户相似度
3. 对 test 轨迹的每个预测任务：
   - 构建原始候选池（用户历史 + 相似用户 + 空间就近，~100 个）
   - **两阶段 LLM 预过滤**：`gpt-5.4` 推断出行意图 → `gpt-5.4` 筛选至 30 个候选
   - `gpt-5.4-mini` 结合画像、上下文、相似用户模式，输出 top-10 排名
4. 用 Acc@1/5/10、MRR 评估预测效果

---

## 数据集结构

```
foursquare_NYC/
├── trips_train.csv        # 7,599 条训练轨迹
├── trips_valid.csv        # 1,000 条验证轨迹
├── trips_test.csv         # 2,698 条测试轨迹
├── user_profile.csv       # 1,055 个用户的统计画像（pipe 分隔）
├── loc2id                 # (lon, lat) → location_id 的 pickle 映射
├── user_index.json        # user_id → [[row_idx, traj_id], ...] 索引
└── prompts_refined/       # 5,764 个 POI 的 markdown 描述文件（{id}.txt）
```

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

LLM 请求现在通过 OpenAI API 调用；代码会优先读取项目根目录下的 `.env`，也兼容直接 `export` 环境变量。本地仍只保留 `bge-m3` 作为嵌入模型。

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
3. 如果存在 `cache/latent_reranker.pt`，使用 learned reranker + prior bank 对候选重排；否则回退到 OpenAI LLM 路径
4. 结果缓存至 `cache/predictions/{traj_id}.json`，**支持断点续传**

可选参数：
```bash
python predict.py --traj-id test_0    # 仅预测单条轨迹（调试用）
python predict.py --dry-run           # 打印 LLM prompt，不调用 API
python predict.py --workers 4         # 并行调用（注意 API rate limit）
python predict.py --mode llm          # 强制走 LLM-only 路径
python predict.py --mode reranker     # 强制走 learned reranker 路径
python predict.py --mode hybrid       # reranker 先缩小候选，再由 LLM 最终排序
python predict.py --forced-include-n 3  # nearby 候选强制保留数量；当前默认配置也是 3
python predict.py --recompute         # 忽略缓存，重新预测所有轨迹
```


### Step 3b - Learned reranker

```bash
python train_reranker.py
```

This trains a latent-intent reranker from train trajectories and builds the prior bank used by `predict.py`. If the reranker cache is present, prediction uses it by default; otherwise it falls back to the OpenAI-based path.

Compared with the earlier version, training now uses multiple prefixes per trajectory, and the reranker features include temporal match signals plus harder negatives from nearby, same-category, and prior-retrieved POIs.

### Step 3 — 评估

```bash
python evaluate.py
```

读取 `cache/predictions/` 中的所有预测结果，与 ground truth 对比，计算并打印：

- **整体指标**：Acc@1、Acc@5、Acc@10、MRR、recall_in_top10
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
├── predict.py             # 入口：Stage 3（预测；优先使用 learned reranker）
├── train_reranker.py      # 入口：训练 latent-intent reranker + prior bank
├── evaluate.py            # 入口：Stage 4（指标评估）
│
├── src/
│   ├── config.py          # 路径、模型名、超参数（统一配置）
│   ├── local_llm.py       # LLM 推理封装（OpenAI + 本地 fallback）
│   ├── data_loader.py     # 数据加载、坐标解析、KD-Tree 空间查询
│   ├── profile_builder.py # Stage 1：gpt-5.4 生成用户画像
│   ├── user_similarity.py # Stage 2：geohash Jaccard + bge-m3 余弦相似度
│   ├── candidate_selector.py  # Stage 3a：原始候选池构建
│   ├── prior_bank.py      # Trajectory prior memory + POI embeddings
│   ├── latent_reranker.py # Learned latent-intent reranker
│   ├── llm_prefilter.py   # Stage 3b：意图推断 + LLM 预过滤（fallback）
│   ├── llm_agent.py       # Stage 3c：gpt-5.4-mini 预测 agent（fallback top-10）
│   ├── evaluator.py       # Stage 4：Acc@K / MRR 计算
│   └── utils.py           # haversine、日志等工具
│
├── models/                # 本地模型权重（huggingface-cli 下载）
│   └── bge-m3/
│
├── cache/
│   ├── profiles/          # {user_id}.json — 用户画像缓存
│   ├── similarity.pkl     # 用户相似度矩阵
│   ├── poi_embeddings.pkl # POI 文本嵌入与统计缓存
│   ├── prior_bank.pkl     # 轨迹前缀 prior memory
│   ├── latent_reranker.pt # 学习式 reranker 权重
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

## 候选召回率分析

由于测试轨迹跨越多天、空间跨度大，候选筛选策略直接影响系统上限：

| 策略 | Recall@N |
|------|----------|
| 纯空间（KD-Tree top-100） | ~37% |
| 纯空间（KD-Tree top-500） | ~52% |
| **历史优先（当前策略，top-100）** | **~51.5%** |

当前策略中 61% 的 ground truth 出现在用户自身的训练历史中，因此优先分配历史访问槽位。Recall@100 ≈ 51.5% 是本系统预测精度的理论上限。
