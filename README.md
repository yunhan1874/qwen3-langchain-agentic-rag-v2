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
| 4 | 原始 Query | 先直接检索；低分时由 Qwen 生成结构化 Query Rewrite，失败时规则降级 |
| 5 | 无 Metadata | chunk 写入 source、line、parent、position、type 等 metadata；两路检索共同过滤 |
| 6 | 无压缩 | 按查询相关句做 extractive context compression，并执行总字符预算 |
| 7 | 无引用约束 | 上下文编号、合法引用编号检查、完整证据及引用缺失标记；不自动补编号 |
| 8 | 无评估 | LCEL 评估器输出 Recall@K、词汇支持代理、句子精确匹配及拒答/引用率；语义 Faithfulness 未评估 |
| 9 | 单次检索 | 有界 Agent 循环按召回结果决定停止、改写重试或拒答 |

## 新版逻辑

```text
docs.txt
-> 按“问题:”还原原始多行问答记录
-> Semantic Chunking
-> Chroma Dense index + BM25 index + Metadata

用户 Query
-> 原始问题 + 显式 Metadata Filter（首轮不调用 LLM 改写）
-> Agent 第 1 次 Hybrid Retrieval
   -> Dense candidates
   -> BM25 candidates
   -> RRF fusion
-> CrossEncoder Reranker
-> Agent 检查证据分数
   -> 足够：停止检索
   -> 不足：Qwen 生成 query + filters，再次检索（总计最多 3 轮）
   -> 仍不足：拒答
-> Context Compression
-> Qwen grounded generation
-> inline citations + structured sources
```

请求生命周期由 LangChain LCEL 主链编排：

```text
prepare_request
| prepare_query（保留原始查询与显式过滤条件）
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

为了安全发布，训练平台密钥和机器绝对路径已改为环境变量。训练最大长度仍为 2048；推理采用独立生成预算：改写最多新增 96 token、回答最多新增 256 token，生成 temperature 仍为 0.1。两种调用共享同一个已加载模型。`agent.py` 与 `agent_rag.py` 仍转到新版 RAG 入口。

### 延迟优化

首轮证据充分时只调用一次 Qwen 生成回答。首轮低分才生成 JSON 改写；后续仍低分时可继续改写，总检索轮数最多 3 次。最后一轮失败后直接拒答；重复查询直接停止。改写 JSON 被 token 上限截断时使用规则降级。

显式传入的 Metadata Filter 从第一轮开始生效，并优先于模型推断的同名字段。模型推断过滤条件仅在低分改写分支生成；如需确保首轮按科室、来源或年份过滤，请显式传入过滤条件。

下面的脚本可对已启动页面连续计时，默认排除一个预热请求，并输出状态、引用数和中位耗时：

```bash
python scripts/benchmark_latency.py --url http://127.0.0.1:17860 --runs 3
```

真实训练模型的 vLLM 后端对比见 [vLLM 测试报告](docs/VLLM_BENCHMARK_RESULTS.md)。固定生成相同 128 token 的测试中，Transformers 中位耗时 3.84 秒，vLLM 为 0.76 秒。服务器展示前端已切换到 vLLM；新安装默认使用 Transformers，可通过环境变量选择后端。

CUDA Graph、编译、Attention 后端、前缀缓存和并发的实测见 [vLLM 消融测试](docs/VLLM_ABLATION_RESULTS.md)，附复现脚本和逐次数据。本次短输入测试中关闭 CUDA Graph 后耗时由 0.765 秒增加到 2.331 秒；各项优化的收益不能直接相加。

项目其余链路的测试见 [优化与可靠性审计](docs/PIPELINE_OPTIMIZATION_AUDIT.md)：包括 BM25 倒排、空过滤预检查、精排候选数、FP16、回答预算和 Qwen 提示词格式，同时记录 Citation、词汇 Faithfulness 代理及索引更新的反例。审计原型未修改正式配置。

审计后的实现已加入 BM25 倒排、显式过滤空匹配预检查、按内容指纹分版本索引、引用状态和未闭合思考块清理；详见 [优化落地与复测](docs/OPTIMIZATION_RELEASE.md)。未启用 FP16、减少候选数或更改提示词格式。

### 使用 vLLM 后端

在独立的 vLLM 环境启动已训练检查点，服务只监听 localhost：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/checkpoint-1082 --served-model-name qwen3-medical \
  --host 127.0.0.1 --port 8001 --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.5 --no-enable-prefix-caching \
  --disable-log-requests --seed 0 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4]}'
```

然后在项目环境中保留原有语料、Embedding、Reranker 和索引配置，导出以下变量并启动前端：

```bash
export RAG_LLM_BACKEND=vllm
export RAG_VLLM_BASE_URL=http://127.0.0.1:8001/v1
export RAG_VLLM_MODEL=qwen3-medical
python -m rag_v2.webapp --default-real
```

客户端使用 raw completion 保留既有提示词格式和采样设置（temperature=0.1、top_p=0.95、top_k=20）。回答最多新增 256 token，改写最多新增 96 token，复用同一个 vLLM 服务。前端不再加载额外的 Qwen 权重；BGE 与 Chroma 仍由项目进程运行。启动时会校验服务中的模型名，服务错误会明确显示，不会自动回退加载 Transformers。

页面状态显示实际后端 `vLLM` 或 `Transformers`。如需回退，在重启前端前设置 `RAG_LLM_BACKEND=transformers`；确认不再有客户端使用 vLLM 后，可停止 vLLM 服务以释放显存。

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

`chunk_id` 可从当前版本的 `rag_v2_index/versions/<指纹>/chunks.jsonl` 获取；相邻的 `manifest.json` 记录语料、Embedding 和切分配置。运行：

```bash
python agent_rag.py --evaluate examples/eval_cases.jsonl --eval-k 3
```

输出：

```json
{"recall@3": 0.0, "faithfulness": null, "faithfulness_method": "not_evaluated", "lexical_support_proxy": 0.0, "exact_sentence_support": 0.0, "case_count": 0, "answered_count": 0, "abstention_rate": 0.0, "citation_rate": 0.0}
```

- `Recall@K`：最终接受的前 K 个 chunk 覆盖相关 chunk 的比例。
- `lexical_support_proxy`：词项覆盖代理，不能检测否定、剂量或语义蕴含。
- `exact_sentence_support`：回答句子与完整证据句子的精确匹配比例；同义改写可能得零分，不代表语义事实支持。
- `faithfulness`：未接入独立语义评估，明确返回 `null`。旧函数 `faithfulness_score` 仅是词汇代理的兼容别名。
- `abstention_rate` 与 `citation_rate`：分别统计拒答率和可回答结果中含合法编号的比例；合法编号不等于事实正确。

支持代理只对 `answered` 的结果计分，使用完整压缩证据；拒答和生成失败不计为高支持回答。评估入口与页面一样调用 LCEL。

示例模板见 `examples/eval_cases.jsonl.example`，需先填入真实 chunk ID。

## 测试

核心测试不下载模型即可运行：

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

它覆盖语义断点、多行记录还原、倒排分数一致性、双路召回、过滤空匹配、Reranker、压缩预算、引用状态、Agent 重试、评估代理、思考块清理、版本缓存和展示模式。

## 配置

| 环境变量 | 用途 | 默认值 |
|---|---|---|
| `QWEN_MODEL_PATH` | Qwen 模型路径或模型 ID | `Qwen/Qwen3-1.7B` |
| `RAG_DOCS_PATH` | 真实模式语料文件 | `./data/docs.txt` |
| `RAG_INDEX_DIR` | Chroma 与 Chunk 索引目录 | `./rag_v2_index` |
| `RAG_EMBEDDING_MODEL` | Dense Embedding | `BAAI/bge-small-zh-v1.5` |
| `RAG_EMBEDDING_REVISION` | 远程 Embedding 模型固定版本，部署建议指定 commit；本地模型会校验配置及权重文件指纹 | 空 |
| `RAG_RERANKER_MODEL` | CrossEncoder Reranker | `BAAI/bge-reranker-base` |
| `RAG_REWRITE_MAX_NEW_TOKENS` | 改写最多新增 token | `96` |
| `RAG_ANSWER_MAX_NEW_TOKENS` | 回答最多新增 token | `256` |
| `RAG_LLM_BACKEND` | 推理后端：transformers / vllm | `transformers` |
| `RAG_VLLM_BASE_URL` | vLLM 本机 API 地址 | `http://127.0.0.1:8001/v1` |
| `RAG_VLLM_MODEL` | 与服务 served-model-name 一致 | `qwen3-medical` |
| `RAG_VLLM_API_KEY` | 可选推理服务认证密钥 | 空 |
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
