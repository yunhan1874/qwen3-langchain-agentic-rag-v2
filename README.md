# Qwen3 医疗 LangChain Agentic RAG v2

这是从原版 `qwen3_langchain_agentic_rag` **复制到独立目录后**建立的新版。原版目录未修改；训练、数据下载、格式转换和原始语料保持不变，只升级 RAG 层。

> 医疗安全声明：本仓库用于 RAG 工程与面试作品集展示，不提供诊断、处方或治疗建议。演示模式只使用仓库内的合成文本。

## 可展示 Agent

项目提供 Gradio Web UI，可观察完整 Agent 决策过程：

- 原始 Query 与 Rewrite 后的 Query
- Metadata Filter
- BM25、Dense、RRF 与 CrossEncoder 分数
- Agent 重试、停止和拒答步骤
- 压缩后的最终证据
- 行内引用与结构化 Citation
- 状态、置信度和最终选择结果

### 30 秒启动演示模式

演示模式不需要下载 Qwen、BGE、Chroma 索引或医疗数据：

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements-demo.txt
python -m rag_v2.webapp --demo-only
```

浏览器打开 `http://127.0.0.1:7860`。

也可以使用 Docker：

```bash
docker build -f Dockerfile.demo -t qwen3-agentic-rag-demo .
docker run --rm -p 7860:7860 qwen3-agentic-rag-demo
```

演示模式会显著标记 `DEMO / SYNTHETIC`，其内容不代表真实医疗知识库输出。

### 使用真实训练模型启动前端

真实模式需要完整 Hugging Face 格式的训练检查点、医疗语料、BGE Embedding
和 CrossEncoder Reranker。通过环境变量指定路径后启动：

```bash
export QWEN_MODEL_PATH=/path/to/checkpoint
export RAG_DOCS_PATH=/path/to/docs.txt
export RAG_INDEX_DIR=/path/to/rag_v2_index
export RAG_EMBEDDING_MODEL=/path/to/bge-small-zh-v1.5
export RAG_RERANKER_MODEL=/path/to/bge-reranker-base

python -m rag_v2.webapp --default-real
```

`--default-real` 会让页面默认选中真实模型模式。服务默认只监听
`127.0.0.1:7860`；远程服务器建议通过 SSH 隧道访问，不要直接暴露医疗语料接口：

```bash
ssh -N -L 17860:127.0.0.1:7860 user@server
```

然后访问 `http://127.0.0.1:17860`。首次运行会执行 Semantic Chunking 并构建
Chroma 索引；后续启动会复用 `RAG_INDEX_DIR`。

## 本版只做的 9 项升级

| # | 原版 | 新版实现 |
|---|---|---|
| 1 | 每行作为固定块 | 相邻句向量相似度断点的 Semantic Chunking，并保留最小/最大长度边界 |
| 2 | Chroma Dense 单路 | BM25 + Chroma Dense，使用 RRF 融合名次 |
| 3 | Top-K 直接进入 LLM | `BAAI/bge-reranker-base` CrossEncoder 重排后只取最终证据 |
| 4 | 原始 Query | Qwen 生成结构化 Query Rewrite；失败时有确定性降级 |
| 5 | 无 Metadata | chunk 写入 source、line、parent、position、type 等 metadata；两路检索共同过滤 |
| 6 | 无压缩 | 按查询相关句做 extractive context compression，并执行总字符预算 |
| 7 | 无引用约束 | 上下文编号、答案内 `[n]` 引用、结构化 `Citation` 与来源清单 |
| 8 | 无评估 | JSONL 评估器输出 Recall@K 与 Faithfulness |
| 9 | 单次检索 | 有界 Agent 循环按召回结果决定停止、改写重试或拒答 |

## 新版逻辑

```text
docs.txt
-> 按“问题:”还原原始多行问答记录
-> Semantic Chunking
-> Chroma Dense index + BM25 index + Metadata

用户 Query
-> Query Rewrite（query + filters）
-> Agent 第 1 次 Hybrid Retrieval
   -> Dense candidates
   -> BM25 candidates
   -> RRF fusion
-> CrossEncoder Reranker
-> Agent 检查证据分数
   -> 足够：停止检索
   -> 不足：改写查询并再次检索（最多 3 步）
   -> 仍不足：拒答
-> Context Compression
-> Qwen grounded generation
-> inline citations + structured sources
```

请求生命周期由 LangChain LCEL 主链编排：

```text
prepare_request
| query_rewrite
| agentic_hybrid_retrieval
| context_compression
| cited_grounded_generation
| return_answer
```

入口的 `LangChainAgenticRAG.invoke()` 只调用 `chain.invoke()`；检索循环的决策依赖每次实际召回的候选数和 Reranker 分数，因此不是把固定流程改名为 Agent。

## 与原版的关系

RAG 的九项升级保持原项目设计。公开展示版没有提交原始医疗数据、生成后的 `docs.txt`、模型权重、Chroma 索引或本地缓存。

以下训练和数据脚本仍保留原始业务用途：

- `data.py`：下载并切分数据
- `docs.py`：生成 RAG 语料
- `train.py`：Qwen3 SFT

为了安全发布，训练平台密钥和机器绝对路径已改为环境变量；最大长度仍为 2048，生成 temperature 仍为 0.1。`agent.py` 与 `agent_rag.py` 仍转到新版 RAG 入口。

## 安装与运行

先按机器的 CUDA/CPU 环境安装合适的 PyTorch，然后安装项目：

```bash
cd qwen3_langchain_agentic_rag_v2
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

复制环境变量模板并按机器情况填写：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

首次运行或语料变化后重建索引：

```bash
python agent_rag.py --rebuild-index
```

单次查询：

```bash
python agent_rag.py --query "高血压患者如何管理血压？"
```

Metadata Filtering（所有条件取交集）：

```bash
python agent_rag.py \
  --query "高血压如何治疗？" \
  --filter document_type=medical_qa \
  --filter source=docs.txt
```

## 评估

评估集每行格式：

```json
{"query":"高血压如何治疗？","relevant_ids":["真实 chunk_id"],"filters":{"document_type":"medical_qa"}}
```

`chunk_id` 可从首次建库生成的 `rag_v2_index/chunks.jsonl` 获取。运行：

```bash
python agent_rag.py --evaluate examples/eval_cases.jsonl --eval-k 3
```

输出：

```json
{"recall@3": 0.0, "faithfulness": 0.0, "case_count": 0}
```

- `Recall@K`：前 K 个重排结果覆盖相关 chunk 的比例。
- `Faithfulness`：回答中的事实陈述有多少能由其引用上下文支持；当前实现是可离线复现的词项覆盖判据。

示例模板见 `examples/eval_cases.jsonl.example`，需先填入真实 chunk ID。

## 测试

核心测试不下载模型即可运行：

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

它覆盖语义断点、多行记录还原、双路召回、Metadata Filtering、Reranker、严格压缩预算、引用校验、Agent 重试、Recall@K、Faithfulness 和展示模式。

## 配置

| 环境变量 | 用途 | 默认值 |
|---|---|---|
| `QWEN_MODEL_PATH` | Qwen 模型路径或模型 ID | `Qwen/Qwen3-1.7B` |
| `RAG_DOCS_PATH` | 真实模式语料文件 | `./data/docs.txt` |
| `RAG_INDEX_DIR` | Chroma 与 Chunk 索引目录 | `./rag_v2_index` |
| `RAG_EMBEDDING_MODEL` | Dense Embedding | `BAAI/bge-small-zh-v1.5` |
| `RAG_RERANKER_MODEL` | CrossEncoder Reranker | `BAAI/bge-reranker-base` |
| `SWANLAB_API_KEY` | 可选训练日志密钥 | 空 |
| `GRADIO_SERVER_NAME` | Web UI 监听地址 | `127.0.0.1` |
| `GRADIO_SERVER_PORT` | Web UI 端口 | `7860` |

## 仓库安全边界

以下内容由 `.gitignore` 排除，不应提交到 GitHub：

- `.env` 与任何真实 API Key
- 原始或格式化训练数据
- 生成后的医疗语料
- 模型权重和缓存
- Chroma 索引
- 训练输出、虚拟环境和缓存

如需公开仓库，请先确认数据集许可，并使用 `rg` 或密钥扫描工具再次检查提交内容。

## 关键技术边界

- `0.28` 是当前实验门槛，不是通用概率，需要用标注验证集校准。
- 当前 Faithfulness 是词项覆盖近似，不是 NLI 或 LLM Judge。
- Citation 保证编号合法和来源可追踪，不等同于事实级蕴含。
- Demo 模式使用 Hashing Embedder、规则改写和合成语料，仅展示流程与可观察性。
