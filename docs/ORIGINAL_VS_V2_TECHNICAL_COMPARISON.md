# Qwen3 医疗 RAG：最初原版与 Agentic RAG v2 完整对比

## 1. 文档范围与结论

本文对比两个实际目录：

- 最初原版：`qwen3_langchain_agentic_rag`
- 当前新版：`qwen3_langchain_agentic_rag_v2`

对比以当前源码为准，不把设计目标当成已实现功能。

一句话概括：

> 原版是“把每个物理文本行向量化，取最相近的 3 条，直接交给 RetrievalQA 生成”；新版是“先恢复完整问答记录并按语义切块，再做 Query Rewrite、Metadata 过滤、BM25 + Dense 混合召回、Reranker、证据门槛、上下文压缩、带引用生成，并由有界 Agent 根据检索反馈决定重试或拒答”。

新版仍以 LangChain 为请求编排主体，但不再使用原版的黑盒式 `RetrievalQA`：

- 原版：LangChain `RetrievalQA.from_chain_type(..., chain_type="stuff")`。
- 新版：LangChain Core LCEL `RunnableLambda` 显式串联每个阶段。
- Chroma、BM25、CrossEncoder 等属于检索数据面，由项目直接控制。

## 2. 核心差异总览

| 维度 | 最初原版 | Agentic RAG v2 | 主要收益 |
|---|---|---|---|
| 文档边界 | `docs.txt` 每个物理行就是一个 Document | 先按 `问题:` 恢复完整问答，再做语义切块 | 避免同一回答被拆成大量失去上下文的行 |
| Chunk 策略 | 行边界固定切块 | 相邻句向量相似度断点 + 120/900 字符边界 | Chunk 更接近语义主题 |
| Embedding | `all-MiniLM-L6-v2` | `BAAI/bge-small-zh-v1.5` | 更适合中文语义检索 |
| 召回 | Chroma Dense 单路 | BM25 + Chroma Dense 双路 | 同时覆盖精确术语和语义改写 |
| 融合 | 无 | Reciprocal Rank Fusion，默认 `k=60` | 消除 BM25 与 cosine 的量纲差异 |
| 候选数量 | Dense Top-3 | Dense 12 + BM25 12 → 融合 12 | 为重排提供更高召回候选池 |
| Reranker | 无 | BGE CrossEncoder，12 候选中重排前 5 | Query 与文档联合编码，提升精排准确性 |
| 最终上下文 | Top-3 无条件进入 LLM | 分数达到 0.28 后最多保留 3 条 | 可以拒绝无可靠证据的问题 |
| Query | 原始问题直接向量化 | Qwen JSON Rewrite + 规则降级 + 检索失败重写 | 补全指代、同义词和检索表达 |
| Metadata | 无 | source、line、title、parent、chunk position、type | 支持检索前过滤与来源追踪 |
| Context | `stuff` 全量拼接 | 相关句抽取、单文档 650 字、总计 1800 字 | 降低无关上下文和 Prompt 成本 |
| Citation | 只在回答后打印 Top-3 文本 | `[n]` 行内引用 + 结构化 `Citation` | 回答事实可以回溯到 chunk |
| 评估 | 无 | Recall@K + 词项覆盖 Faithfulness | 可以对检索和回答进行回归比较 |
| 控制流程 | 单次检索 | 最多 3 步的检索 Agent | 根据真实检索结果决定停止、重试或拒答 |
| 编排方式 | LangChain `RetrievalQA` | LangChain LCEL 显式节点图 | 每个阶段可测试、可观测、可替换 |

## 3. 哪些部分没有改变

此次升级只改 RAG 层。以下文件已经逐字节核对，与原版一致：

- `data.py`
- `docs.py`
- `train.py`
- `train.jsonl`
- `val.jsonl`
- `train_format.jsonl`
- `val_format.jsonl`
- `docs.txt`

因此下面这些训练行为也没有改变：

- 数据仍来自 ModelScope `krisfu/delicate_medical_r1_data`。
- 仍以 `random.seed(42)` 打乱后按行做 90%/10% 切分。
- 仍将 `think` 包装进 `<think>...</think>`，与答案一起作为监督目标。
- 仍使用 Qwen3-1.7B、最大长度 2048。
- 仍是 Transformers `Trainer` 全参数训练，没有 LoRA/PEFT。
- 仍使用 batch size 1、gradient accumulation 4、2 epochs、learning rate `1e-4`。
- 仍使用 gradient checkpointing、BF16 和 SwanLab。
- RAG 语料仍来自训练集格式化文件的“问题 + 输出”。

这意味着新版不是一次训练系统重构。原版在数据质量、PII、数据泄漏、训练凭据管理以及把 `<think>` 内容写入 RAG 语料等方面的风险仍然存在。

## 4. 最初原版的完整逻辑

### 4.1 离线数据与训练链路

```text
ModelScope 医疗数据集
-> 全量转成 Python list
-> random.seed(42) 后随机打乱
-> 按行 90% train / 10% validation
-> question / think / answer
-> instruction / input / output
   output = <think>{think}</think> + answer
-> 手工拼接 ChatML token
-> instruction token 的 label 设为 -100
-> response token 参与 loss
-> 超过 2048 token 直接截断
-> Qwen3-1.7B 全参数 SFT
-> 每 100 steps 验证、每 400 steps 保存
-> 验证集前 3 条做主观生成检查
```

训练目标可以写成：

```text
input_ids = system + user + assistant_prefix + response + pad
labels    = -100   × instruction_length
            + response_ids
            + pad_id
```

只有 response 区域参与交叉熵损失。这里的 response 同时包含 `<think>` 与最终答案。

### 4.2 原版语料生成

`docs.py` 把每条训练样本转成：

```text
问题: {input}
答案: {output}
```

但写入 `docs.txt` 时，`output` 本身可能包含多行。原版查询脚本随后使用：

```python
for line in f:
    docs.append(line.strip())
```

因此原本的一条问答会再次被拆成许多物理行。原版所谓 Document 并不是“一条完整医学问答”，而是“一个非空文本行”。

### 4.3 原版建库逻辑

```text
docs.txt
-> 逐物理行读取
-> 精确字符串去重
-> all-MiniLM-L6-v2 embedding
-> Chroma.from_texts(..., persist_directory="chroma_db")
```

原版没有：

- parent document ID；
- chunk position；
- source、title、category 等 metadata；
- embedding 模型与语料版本绑定；
- collection 数量或完整性检查；
- 显式分批写入；
- 语义边界。

### 4.4 原版在线问答逻辑

```text
用户原始 Query
-> all-MiniLM-L6-v2 Query embedding
-> Chroma similarity search
-> 固定 Top-3
-> RetrievalQA stuff chain
-> Top-3 全文直接拼入 Prompt
-> Qwen3 生成
-> 对回答文本按行去重
-> 额外打印 Top-3 原始文档前 500 字符
```

原版最重要的逻辑特点是：

> 相似度检索总能排出第一、第二、第三名，但“排名靠前”不代表“绝对相关”。

即使用户询问知识库完全没有覆盖的主题，Chroma 仍会返回最接近的 3 条。原版没有 Reranker、绝对证据门槛或拒答状态，因此这些文档会被无条件送入 LLM。

### 4.5 原版的优点

- 代码短，适合快速验证 RAG 是否能跑通。
- 直接使用 LangChain 和 Chroma，学习成本低。
- 本地 Qwen、embedding、向量库、交互 CLI 构成了完整 demo。
- `return_source_documents=True` 至少能够展示检索结果。

### 4.6 原版的主要问题

- 物理行切分破坏问答内部语义。
- 单路 Dense 对药品名、指标名、缩写和精确术语不稳定。
- 固定 Top-3 无条件进入生成。
- 没有 Query Rewrite 和多次检索。
- 没有 Metadata Filtering。
- 没有 Context Compression。
- 打印来源不等于回答事实与来源绑定。
- 没有检索与生成评估。
- 没有证据不足拒答。

## 5. 新版的完整逻辑

### 5.1 新版离线索引链路

```text
原始 docs.txt（内容保持不变）
-> 遇到“问题:”时识别新问答记录
-> 恢复 2166 条多行问答记录
-> 完整记录去重
-> 添加 document metadata
-> 中文标点/换行切句
-> BGE 生成每个句子的归一化向量
-> 计算相邻句 cosine similarity
-> 低相似度语义断点 + 长度强制断点
-> 生成稳定 chunk_id 和位置 metadata
-> chunks.jsonl
-> Chroma cosine collection（每批 256 条）
-> 同一批 chunks 建立内存 BM25 索引
```

每个 chunk 至少包含这些字段：

```text
document_id / chunk_id
parent_document_id
source / source_path / source_line
title / document_type
chunk_index
start_sentence / end_sentence
text
```

### 5.2 新版在线 LCEL 主链

```text
prepare_request
-> query_rewrite
-> agentic_hybrid_retrieval
-> context_compression
-> cited_grounded_generation
-> return_answer
```

生产入口 `LangChainAgenticRAG.invoke()` 最终执行：

```python
self.chain.invoke({"query": query, "filters": filters or {}})
```

LangChain LCEL 是请求控制面；BM25、Chroma 和 CrossEncoder 是检索数据面。

### 5.3 新版单次 Agent 尝试内部逻辑

```text
rewritten query + metadata filters
        |
        +-> Chroma Dense Top-12
        |
        +-> BM25 Top-12
                    |
                    v
                RRF 融合
                    |
              fused Top-12
                    |
           CrossEncoder Reranker
                    |
              reranked Top-5
                    |
          top score >= 0.28 ?
             /              \
           是                否
       接受最多 3 条      生成下一检索 Query
                              |
                       最多总计 3 次尝试
```

### 5.4 新版终止逻辑

Agent 有三类终止条件：

1. `evidence_sufficient`：存在候选，最高 Reranker 分数达到门槛。
2. `duplicate_query`：下一次改写与已经尝试的 Query 重复，避免死循环。
3. 达到 `max_agent_steps=3`：保留各轮中最高分候选；仍未达到门槛则拒答。

证据不足时返回：

```text
status = abstained
citations = []
answer = 当前知识库没有检索到足够可靠的证据，无法据此回答。
```

证据充分时才进入压缩和生成。

## 6. 九项升级的详细技术点

### 6.1 固定切块 → Semantic Chunking

#### 原版

原版不是严格意义上的“固定 N 字符切块”，而是更粗糙的“每个物理行一个块”。同一条问答的推理和答案可能被拆开。

#### 新版

先把 `docs.txt` 恢复为完整问答记录，然后对每个记录切句。设相邻句向量为 `e_i` 和 `e_(i+1)`：

```text
similarity_i = cosine(e_i, e_(i+1))
             = (e_i · e_(i+1)) / (||e_i|| ||e_(i+1)||)
```

将该文档所有相邻句相似度的第 25 百分位作为断点阈值：

```text
breakpoint = percentile(similarities, 25)
```

当满足以下条件时切分：

```text
similarity_i <= breakpoint
and current_chunk_chars >= 120
```

同时有强制上限：

```text
projected_chunk_chars > 900 -> 强制切分
```

收益：

- 相同主题句尽量保留在一起。
- 主题跳变处优先切分。
- 最小长度防止碎片过多。
- 最大长度避免上下文块过大。
- parent/chunk/position metadata 使切块可追踪。

边界：语义断点是文档内部相对阈值，不是全局学习出来的最佳阈值；若一条记录只有一个句子，就不会发生语义切分。

### 6.2 单纯 Dense → BM25 + Dense Hybrid Retrieval

#### Dense 路

- 模型：`BAAI/bge-small-zh-v1.5`。
- 文档与 Query embedding 均归一化。
- Chroma 使用 cosine 空间。
- 默认召回 `dense_k=12`。

Dense 擅长：

- 同义词；
- 语义改写；
- 用户描述与知识库表达不完全一致的情况。

#### BM25 路

中文分词采用轻量规则：

- 中文单字 unigram；
- 相邻中文 bigram；
- 保留英文、数字、药品缩写和带符号的 Latin token。

BM25 公式为：

```text
score(D, Q) = Σ IDF(q_i) ×
              f(q_i,D) × (k1 + 1)
              ------------------------------------------
              f(q_i,D) + k1 × (1 - b + b × |D| / avgdl)
```

当前参数：

```text
k1 = 1.5
b  = 0.75
bm25_k = 12
```

BM25 擅长：

- 药物名称；
- 检验指标；
- 疾病缩写；
- 精确数字和罕见术语。

#### RRF 融合

BM25 分数和 cosine 分数不在同一量纲，不能直接相加。新版使用 Reciprocal Rank Fusion：

```text
RRF(d) = Σ_r 1 / (k + rank_r(d))
```

其中：

```text
k = 60
r ∈ {dense, bm25}
```

代码再用理论最大值归一化：

```text
max_RRF = 2 / (60 + 1)
fusion_score = min(1, RRF / max_RRF)
```

RRF 的职责只是把候选排成一个更稳的顺序，不被某一路原始分数尺度支配。

### 6.3 Top-K 直接喂 LLM → Reranker

原版 Dense Top-3 直接进入 `stuff` chain。新版先扩大召回，再缩小精排：

```text
Dense 12 + BM25 12
-> RRF fused 12
-> CrossEncoder rerank 5
-> evidence gate 后最终最多 3
```

Bi-Encoder Dense 检索分别编码 Query 和文档，适合大规模召回；CrossEncoder 把 `(query, document)` 一起编码，计算量更大，但能建模细粒度交互，适合小候选集精排。

CrossEncoder 原始输出先映射到 `[0,1]`：

```text
若原始分已在 [0,1]：直接使用
否则：sigmoid(clamp(raw_score, -30, 30))
```

最终分数：

```text
rerank_score = 0.85 × cross_encoder_score
             + 0.15 × fusion_score
```

默认门槛：

```text
min_relevance_score = 0.28
```

注意：`0.28` 是工程初始值，不是通用概率。正式上线前应在标注验证集上做阈值扫描和校准。

### 6.4 原始 Query → Query Rewrite

新版让现有 Qwen 输出：

```json
{"query":"适合知识库检索的表达","filters":{}}
```

改写目标：

- 补全指代；
- 加入医学同义表达；
- 去掉无助于检索的对话前缀；
- 保留原始意图；
- 仅在用户明确表达条件时提取 Metadata filter。

允许模型产生的过滤字段有白名单：

```text
source / source_line / document_type
department / category / year
```

命令行显式传入的 filter 会覆盖 LLM 推断结果：

```text
merged_filters = inferred_filters + explicit_filters
```

如果模型没有返回合法 JSON，则使用规则降级：去除“请问”“我想知道”“医生您好”等前缀并规范空白。

当检索分数不足时，Agent 把“没有候选”或“最高相关度仅 X”作为反馈，再请求一次检索 Query。模型调用失败则退回原始 Query 或去标点 Query。

### 6.5 无 Metadata → Metadata Filtering

文档加载时生成：

```text
source
source_path
source_line
title
document_type
```

Semantic Chunking 后追加：

```text
parent_document_id
chunk_id
chunk_index
start_sentence
end_sentence
```

过滤不是在召回后再删除：

- Dense 路把条件转换成 Chroma `where`，在向量检索阶段下推。
- BM25 路在计算每个文档分数前检查相同 filters。
- 多个条件使用 AND。
- list/tuple/set 值使用成员匹配，Chroma 对应 `$in`。

这样可以避免错误科室、错误来源或错误年份的文档先占满 Top-K。

当前原始语料只天然具有 source、line、title 和 `document_type=medical_qa`。`department/category/year` 只有在以后摄取数据时实际写入 metadata 才能有效过滤。

### 6.6 无 Context Compression → Extractive Compression

Reranker 决定保留哪些 chunk，Compressor 再决定每个 chunk 中保留哪些句子。

步骤：

1. 将 Rewrite 后的 Query 分词。
2. 将候选 chunk 切句。
3. 按“句子与 Query 的 token 交集数量”降序排列。
4. 在单文档 650 字符预算内选择句子。
5. 按原始句子位置恢复顺序。
6. 对所有文档再执行总上下文 1800 字符预算。
7. 保留原 chunk ID 和 metadata，并标记 `compressed=True`。

这是 extractive compression，不会改写证据，因此引用仍可以回溯到原 chunk。

边界：词项交集不是语义压缩模型；同义表达可能得分低。若首句本身超过预算，会直接截断到预算长度。

### 6.7 无 Citation → 来源引用

新版把最终上下文编号：

```text
[1] source=docs.txt; chunk_id=...
证据文本

[2] source=docs.txt; chunk_id=...
证据文本
```

生成 Prompt 要求每个医学事实使用 `[n]` 引用。输出后还有校验：

- 删除不存在的引用编号；
- 如果存在证据但模型没有任何引用，至少补上 `[1]`；
- 同时返回结构化 `Citation`：citation_id、chunk_id、source、title、excerpt。

最终 `RAGAnswer` 还包含：

```text
answer
citations
original_query
rewritten_query
status
agent_steps
retrieved_ids
confidence
```

重要边界：自动补 `[1]` 只能保证“格式上有来源”，不能证明整句话真的被来源支持。事实级支持仍需要 Faithfulness 评估或更强的 entailment/LLM judge。

### 6.8 无评估 → Recall@K + Faithfulness

#### Recall@K

```text
Recall@K = |TopK_retrieved ∩ Relevant| / |Relevant|
```

它回答：标注为相关的 chunk，有多少出现在系统返回的前 K 条中。

当前实现使用最终 `RAGAnswer.retrieved_ids`，即经过 Agent 门槛并限制到 `final_k` 后的 IDs。因此它评估的是“最终被接受证据的召回”，而不是纯召回层的候选 Recall。默认 `final_k=3`，当评估 `K>3` 时也不会得到超过 3 条结果。

如果要单独测召回器，后续应直接对 RRF 或 Reranker 输出计算 Recall@K。

#### Faithfulness

当前实现是可离线复现的词项支持度：

1. 去掉答案中的 `[n]`。
2. 将答案切成 claims。
3. 计算每个 claim token 在引用 excerpt 中的覆盖率。
4. 覆盖率达到 0.35，视为该 claim 被支持。

```text
claim_coverage = |tokens(claim) ∩ tokens(context)| / |tokens(claim)|

Faithfulness = supported_claims / all_claims
```

优点：无需额外 Judge 模型、结果可重复。

局限：

- 词项重合不等于逻辑蕴含；
- 同义改写可能被低估；
- 复制证据但改变否定词可能被高估；
- 当前拒答答案没有引用上下文，也可能得到较低分。

### 6.9 单次检索 → Agentic RAG

这里的 Agentic 不等于开放式自主 Agent，而是一个安全的有界检索决策循环。

Agent 每轮观察：

- 是否有候选；
- 最高 Reranker 分数；
- 下一 Query 是否重复；
- 已使用多少步。

可执行动作：

```text
STOP_AND_ANSWER
REWRITE_AND_RETRY
ABSTAIN
```

控制逻辑：

```python
for attempt in range(max_agent_steps):
    retrieve()
    rerank()
    if top_score >= threshold:
        stop
    query = retry_query(feedback)

if no accepted evidence:
    abstain
```

它之所以比固定 pipeline 更接近 Agent，是因为下一步动作取决于上一轮工具结果；它之所以可控，是因为动作集合固定、最多 3 步，并且有重复 Query 终止条件。

## 7. 两版逻辑并排展示

### 7.1 原版

```text
数据集
-> 随机 90/10 split
-> Qwen3 全参数 SFT
-> 训练集 question + think + answer 写入 docs.txt
-> 每个物理行作为 Document
-> MiniLM embedding
-> Chroma 单路 similarity search Top-3
-> RetrievalQA stuff
-> Qwen 生成
-> 回答按行去重
-> 额外显示来源文本
```

### 7.2 新版

```text
相同数据集、相同 split、相同训练、相同 docs.txt
-> 按“问题:”恢复完整问答
-> BGE 相邻句语义断点切块
-> chunk metadata + stable chunk ID
-> Chroma cosine index + BM25 index

用户问题
-> LCEL prepare_request
-> Qwen Query Rewrite + filters
-> Agent Loop
   -> Dense Top-12（filter pushdown）
   -> BM25 Top-12（相同 filter）
   -> RRF fused Top-12
   -> CrossEncoder Top-5
   -> 证据门槛 0.28
   -> 不足则重写并重试，最多 3 步
-> 最多接受 3 条
-> 每条压缩到 650 字，总上下文 1800 字
-> 编号证据
-> Qwen grounded generation
-> 引用编号校验
-> RAGAnswer + Citation + trace steps
-> 可离线计算 Recall@K / Faithfulness
```

## 8. 架构与代码对应关系

| 职责 | 原版 | 新版 |
|---|---|---|
| CLI/主入口 | [`agent_rag.py`](../../qwen3_langchain_agentic_rag/agent_rag.py) | [`agent_rag.py`](../agent_rag.py)、[`cli.py`](../rag_v2/cli.py) |
| LangChain 编排 | `RetrievalQA` | [`agent.py`](../rag_v2/agent.py) 中的 LCEL graph |
| 文档加载与切块 | 主入口逐行读取 | [`semantic_chunking.py`](../rag_v2/semantic_chunking.py) |
| Embedding | LangChain HuggingFaceEmbeddings | [`embeddings.py`](../rag_v2/embeddings.py) |
| Dense 检索 | LangChain Chroma retriever | [`retrieval.py`](../rag_v2/retrieval.py) |
| BM25/RRF | 无 | [`retrieval_base.py`](../rag_v2/retrieval_base.py) |
| Reranker | 无 | [`reranker.py`](../rag_v2/reranker.py) |
| Query Rewrite | 无 | [`query_rewrite.py`](../rag_v2/query_rewrite.py) |
| Context Compression | 无 | [`compression.py`](../rag_v2/compression.py) |
| Citation | 仅打印 source documents | [`citations.py`](../rag_v2/citations.py) |
| Grounded Prompt | RetrievalQA 默认 chain | [`generation.py`](../rag_v2/generation.py) |
| 输出协议 | dict | [`models.py`](../rag_v2/models.py) 的 `RAGAnswer` |
| 评估 | 无 | [`evaluation.py`](../rag_v2/evaluation.py) |
| 参数管理 | 文件顶部常量 | [`config.py`](../rag_v2/config.py) |

## 9. 新版默认参数及含义

| 参数 | 默认值 | 含义 |
|---|---:|---|
| embedding model | `BAAI/bge-small-zh-v1.5` | 中文 Dense embedding |
| reranker model | `BAAI/bge-reranker-base` | CrossEncoder 精排 |
| semantic breakpoint | 25 percentile | 文档内低相似度断点 |
| minimum chunk | 120 chars | 防止碎片块 |
| maximum chunk | 900 chars | 防止过大块 |
| dense_k | 12 | Dense 候选数 |
| bm25_k | 12 | BM25 候选数 |
| fused_k | 12 | RRF 后候选数 |
| rerank_k | 5 | CrossEncoder 后保留数 |
| final_k | 3 | 最终进入上下文的最大证据数 |
| RRF constant | 60 | 平滑 rank 差异 |
| relevance gate | 0.28 | 最低 Reranker 接受分 |
| per-document compression | 650 chars | 单个证据压缩预算 |
| total context | 1800 chars | 总证据预算 |
| agent steps | 3 | 最大检索尝试次数 |
| generation max length | 2048 | 保持原版配置 |
| temperature | 0.1 | 保持原版配置 |

## 10. 典型行为对比

### 10.1 用户使用精确药名

问题：

```text
盐酸甲氧明注射液如何测定含量？
```

原版：

- 依赖 MiniLM 是否能把中文药名和文档向量拉近。
- 只返回 Dense Top-3。

新版：

- BM25 利用“盐酸甲氧明注射液”“含量”等精确 token。
- Dense 同时利用语义表达。
- RRF 提升双路都命中的证据。
- CrossEncoder 判断问法与证据是否真正匹配。

### 10.2 用户使用口语或指代

问题：

```text
医生您好，这个药平时应该怎么吃？
```

原版：

- 原始 Query 直接检索，“这个药”没有明确实体。

新版：

- Query Rewrite 尝试补全对话上下文并去掉寒暄。
- 如果没有可补全的上文，仍可能召回不足并拒答；系统不会凭空知道“这个药”是什么。

### 10.3 用户问知识库外问题

原版：

- Chroma 仍会返回最近的 3 条。
- RetrievalQA 仍会生成。

新版：

- 第一轮低分后尝试改写。
- 最多 3 次仍未达到 0.28，返回 `abstained`。
- 不生成引用，不把低可信候选交给 Qwen。

## 11. 已验证结果

当前新版完成过以下验证：

- 10 项单元/集成测试通过。
- LangChain Core 1.5.3 `chain.invoke()` 通过。
- LCEL Mermaid graph 导出通过。
- Python 3.10 使用 `typing_extensions.TypedDict` 的兼容边界已覆盖。
- Chroma 1.5.9 真实持久化 smoke test 通过。
- 8 条真实问答恢复后产生 42 个 semantic chunks。
- 42 个向量成功写入 Chroma。
- “盐酸甲氧明注射液如何测定含量”正确召回目标问答。
- 600 条模拟 corpus 验证 256/256/88 分批写入。
- Python wheel 成功构建为 `qwen3_langchain_agentic_rag_v2-2.0.0`。
- 原版非 RAG 文件与新版对应文件逐字节一致。

当前没有完成的验证：

- 当前备份环境的默认 Qwen 路径不存在，因此没有完成真实 Qwen 生成端到端测试。
- BGE 缓存不完整，Hugging Face TLS 下载失败，因此真实 smoke test 的 Chroma 路径使用了确定性测试 embedding；BGE 接口与切块算法由单元测试覆盖，但仍应在目标运行环境补一次全量建库。

## 12. 不能夸大的边界

新版显著增强了检索链路，但还不是完整的医疗生产系统：

- 没有急症识别与人工升级流程。
- 没有医学免责声明策略和区域化急救信息。
- 没有 PII 脱敏、Prompt Injection 检测或文档信任等级。
- 没有 corpus checksum、embedding model fingerprint 或完整原子索引发布。
- Chroma 当前主要通过向量数量检查半成品索引，不验证每个 ID 和语料哈希。
- Reranker 门槛尚未通过大规模标注集校准。
- Query Rewrite 与回答共用 Qwen，增加模型调用延迟。
- Citation 校验主要保证编号合法，不等于事实级蕴含证明。
- Faithfulness 是词项覆盖近似，不是 NLI 或 LLM-as-a-Judge。
- 当前 Recall@K 更接近“门禁后的最终证据召回”，不是纯 Retriever Recall。
- `docs.txt` 的记录恢复依赖每条记录以 `问题:` 开头。
- 原训练脚本中的凭据、全参数训练和 `<think>` 语料策略保持原样，未在本次升级范围内修复。

## 13. 面试时如何讲这次升级

### 13.1 30 秒版本

> 原版是一个最小可运行 RAG：把 docs.txt 每行做 MiniLM embedding，Chroma 取 Top-3，再用 RetrievalQA stuff 生成。它的问题不是不能跑，而是没有绝对相关性判断，知识库外问题也会强行返回 3 条。新版保持训练和语料不变，只重构 RAG：恢复完整问答后做 BGE Semantic Chunking，使用 BM25 + Dense + RRF 扩大召回，CrossEncoder 精排并通过 0.28 门槛决定回答或拒答；前面加入 Query Rewrite 和 Metadata Filter，后面加入 Context Compression、结构化 Citation、Recall@K 和 Faithfulness，并用最多 3 步的 LCEL Agent 根据召回反馈重试。

### 13.2 为什么不是直接把 BM25 分数和 cosine 相加

> 两者量纲和分布不同，直接加权需要校准。RRF 只使用名次，先得到稳定候选顺序；随后 CrossEncoder 独立精排，避免把融合排序分误当成相关概率。

### 13.3 为什么先召回 12 条再重排到 3 条

> 召回阶段目标是高 Recall，精排阶段目标是高 Precision。若 Dense 一开始只取 3 条，BM25 和 Reranker 没有机会纠正漏召回；若把 12 条全交给 LLM，上下文噪声和成本又会上升。

### 13.4 为什么称为 Agentic RAG

> 因为系统不是预先固定只检索一次。每轮会读取候选数量和 Reranker 分数，再选择停止、改写 Query 重试或拒答；同时用 3 步上限和重复 Query 检测控制风险。这是有界 Agent，而不是开放式自主规划。

### 13.5 新版最重要的失败控制

> 最重要的不是回答更多，而是在证据不足时不回答。原版只有相对排名，新版增加了 CrossEncoder 绝对门槛和显式 `abstained` 状态，低分候选不会进入生成。

## 14. 运行与评估

安装：

```bash
cd /root/codex_workspace/qwen3_langchain_agentic_rag_v2
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

首次建库：

```bash
python agent_rag.py --rebuild-index --model <实际 Qwen3 模型目录>
```

单次查询：

```bash
python agent_rag.py \
  --query "高血压患者如何管理血压？" \
  --filter document_type=medical_qa
```

评估 JSONL：

```json
{"query":"高血压如何治疗？","relevant_ids":["真实 chunk_id"],"filters":{"document_type":"medical_qa"}}
```

运行：

```bash
python agent_rag.py --evaluate <eval_cases.jsonl> --eval-k 3
```

## 15. 最终结论

原版解决的是“能不能把本地 Qwen、Chroma 和 LangChain 串起来”；新版解决的是“检索结果是否可靠、是否应该回答、上下文是否干净、答案能否回溯、系统能否评估”。

两者不是训练模型能力的对比，而是 RAG 工程成熟度的对比：

```text
原版：Demo-oriented RAG
新版：可测试、可解释、可拒答、可评估的有界 Agentic RAG
```
