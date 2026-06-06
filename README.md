# NextPOI — LLM-based Next POI Prediction

三阶段可解释流水线：用户画像构建 → 候选池筛选 + LLM 排序 → 指标评估。

```text
build_profiles.py  ->  predict_v2.py  ->  evaluate_v2.py
 user grounding         llm-only        metrics
```

核心设计：

1. 用 `train + valid` 轨迹为每个用户构建叙事性出行画像。
2. 用用户画像嵌入和空间访问重叠计算相似用户。
3. 对每个测试轨迹，先构建候选池（v2: quota-based），再做时间感知的 LLM 意图推断与候选缩减，最后由 LLM 输出 top-10 排名。
4. 用 `Hit@K / N@K / MRR` 评估整体与分层表现。

---

## 数据集

支持 `nyc`、`ca`、`tky` 三个数据集，默认使用 NYC。

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
│   ├── trips_train.csv / trips_valid.csv / trips_test.csv
│   ├── loc2id / user_index.json / prompts_refined/
│   └── ...
└── tky/
    └── (同上结构)
```

轨迹数据格式：每条记录包含 `user_id`、`traj_id`，以及由 `"lon,lat,timestamp"` 字符串组成的 checkin 列表。测试集预测任务定义为：给定一条轨迹的前 N-1 个 checkin，预测第 N 个位置。

---

## 环境配置

```bash
pip install pandas scipy geohash2 tqdm
pip install torch transformers accelerate bitsandbytes
pip install huggingface_hub FlagEmbedding openai
```

**API 密钥**：

```bash
# 项目根目录 .env
cat > .env <<'EOF'
OPENAI_API_KEY=your_key_here
# OPENAI_BASE_URL=https://your-proxy-or-compatible-endpoint/v1
EOF
```

LLM 请求通过 OpenAI API 调用；代码优先读取 `.env`，也兼容 `export` 环境变量。

**本地嵌入模型**（用于用户相似度）：

```bash
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3
```

---

## 使用流程

### Step 1 — 构建用户画像

```bash
python build_profiles.py
```

- 调用 LLM 为每位用户生成叙事性出行画像，缓存至 `cache/<dataset>/profiles/{user_id}.json`
- 使用 **bge-m3** 对画像做文本嵌入，结合 geohash Jaccard 相似度，构建用户相似度矩阵 → `cache/<dataset>/similarity.pkl`
- **支持断点续传**：已有缓存自动跳过
- **支持多线程**：`--workers` 并发发起 LLM 请求

```bash
python build_profiles.py --user-id 1              # 仅构建单个用户（调试）
python build_profiles.py --recompute              # 忽略缓存，全量重建
python build_profiles.py --workers 16             # 并行构建（注意 API rate limit）
python build_profiles.py --skip-similarity        # 只构建画像，跳过相似度
python build_profiles.py --build-transitions      # 从 train+val 挖掘转移概率
```

### Step 2 — 预测下一个 POI

```bash
python predict_v2.py
```

v2 流水线流程：

1. 读取用户画像和相似度矩阵
2. **Quota-based 候选池构建**（`candidate_selector_v2.py`）：历史访问配额 + 近邻探索配额
3. LLM 意图推断 + 候选预过滤（`llm_prefilter_v2.py`）
4. LLM 最终排序 → top-10 结果
5. 结果缓存至 `cache/<dataset>/predictions_v2/{traj_id}.json`，**支持断点续传**

```bash
python predict_v2.py --traj-id test_0     # 仅预测单条轨迹（调试）
python predict_v2.py --dry-run            # 打印 LLM prompt，不调用 API
python predict_v2.py --workers 4          # 并行调用（注意 API rate limit）
python predict_v2.py --recompute          # 忽略缓存，全量重新预测
python predict_v2.py --dataset ca         # 切换到其他数据集
```

### Step 3 — 评估

```bash
python evaluate_v2.py
```

读取 `cache/<dataset>/predictions_v2/` 中的预测结果，计算：

- **整体指标**：Hit@1、Hit@5、Hit@10、N@1、N@5、N@10、MRR
- **分层指标**：按用户数据丰富程度、轨迹长度、目标时段分别统计

结果保存至 `results/<dataset>/evaluation_results.json`。

```bash
python evaluate_v2.py --dataset ca                          # 切换数据集
python evaluate_v2.py --predictions-dir /path/to/predictions # 自定义预测目录
python evaluate_v2.py --limit 500                            # 仅评估前 N 条
```

---

## 推荐命令

```bash
# 完整流水线
python build_profiles.py --build-transitions
python build_profiles.py --workers 16
python predict_v2.py --workers 4
python evaluate_v2.py
```

切换数据集：

```bash
export NEXTPOI_DATASET=ca
# 或
python predict_v2.py --dataset ca
```

---

## 项目结构

```
NextPOI/
├── build_profiles.py         # Stage 1+2：用户画像 + 相似度矩阵
├── predict_v2.py             # Stage 3：v2 预测流水线（quota-based）
├── evaluate_v2.py            # Stage 4：指标评估
│
├── src/
│   ├── config.py             # 路径、模型名、超参数（统一配置）
│   ├── data_loader.py        # 数据加载、坐标解析、KD-Tree 空间查询
│   ├── profile_builder.py    # Stage 1：LLM 生成用户画像
│   ├── user_similarity.py    # Stage 2：geohash Jaccard + bge-m3 余弦相似度
│   ├── candidate_selector.py     # v1 候选池构建
│   ├── candidate_selector_v2.py  # v2 候选池构建（quota-based）
│   ├── llm_prefilter.py          # v1 LLM 意图推断 + 预过滤
│   ├── llm_prefilter_v2.py       # v2 LLM 意图推断 + 预过滤
│   ├── llm_agent.py          # LLM 预测 agent（最终排序）
│   ├── local_llm.py          # LLM API 调用封装
│   ├── evaluator.py          # Hit@K / MRR 计算
│   ├── embedding_utils.py    # 文本嵌入工具
│   ├── utils.py              # haversine、日志等工具
│   └── __init__.py
│
├── models/                   # 本地模型权重
│   └── bge-m3/
│
├── cache/<dataset>/
│   ├── profiles/             # {user_id}.json — 用户画像缓存
│   ├── similarity.pkl        # 用户相似度矩阵
│   ├── intent/               # 意图推断缓存
│   ├── prefilter/            # 预过滤结果缓存
│   ├── predictions/          # v1 预测结果
│   ├── predictions_v2/       # v2 预测结果
│   └── pools_v2/             # v2 候选池缓存
│
└── results/<dataset>/
    └── evaluation_results.json
```

---

## LLM 配置

所有模型配置在 `src/config.py` 中：

```python
PROFILE_LLM_MODEL    = "gpt-5.4"        # 用户画像构建（离线，一次性）
PREDICTION_LLM_MODEL = "gpt-5.4-mini"   # 最终 POI 排序
INTENT_LLM_MODEL     = "gpt-5.4"        # 出行意图推断
PREFILTER_LLM_MODEL  = "gpt-5.4"        # 候选预过滤
EMBEDDING_MODEL      = "BAAI/bge-m3"    # 画像文本嵌入（本地）
```

| 用途 | 调用次数 |
|------|---------|
| 用户画像构建 | ~1,055 次（一次性离线） |
| 出行意图推断 | ~2,698 次 |
| 候选预过滤 | ~2,698 次 |
| 最终 POI 排序 | ~2,698 次 |
| 画像嵌入 | ~1,055 次（一次性离线） |

---

## 候选召回率分析

由于测试轨迹跨越多天、空间跨度大，候选筛选策略直接影响系统上限：

| 策略 | Recall@100 |
|------|-----------|
| 纯空间（KD-Tree top-100） | ~37% |
| 纯空间（KD-Tree top-500） | ~52% |
| **历史优先 + 空间探索（v1）** | **~51.5%** |

61% 的 ground truth 出现在用户自身的训练历史中，因此 v2 采用 quota-based 策略（`QUOTA_HISTORY=27, QUOTA_NEARBY=3`），在历史覆盖与空间探索之间取得平衡。Recall@100 是本系统预测精度的理论上限。
