# Qwen3 真实训练模型：Transformers 与 vLLM 对比

测试日期：2026-10-26。结论：在此次单请求测试中，vLLM 固定生成 128 token 的中位耗时约为 Transformers 的五分之一。性能对比时 Gradio 页面尚未切换，临时测试服务在对比后已停止。后续用户已要求切换，正式展示前端已接入独立的 vLLM 服务，接入方式见 README；下列数据仍为当时的对比测试记录。

## 条件

- GPU：RTX 4090，24 GB；驱动 580.178.04。
- 模型：同一份用户训练的 Qwen3-1.7B `checkpoint-1082` 完整权重，BF16。
- Transformers：4.57.6，PyTorch 2.5.1+cu121，现有项目环境。
- vLLM：0.11.0，Transformers 4.57.6，PyTorch 2.8.0+cu128，独立测试环境。
- RAG 客户端、BGE Embedding、BGE CrossEncoder、2,502 个 Chroma chunk、召回与压缩配置相同。
- 两侧均使用贪心解码；自然回答最多新增 256 token，固定长度测试强制生成 128 token。
- 使用原有生成器构造的原始文本 prompt 调用 completion；不新增聊天模板或关闭思考指令，以避免改变提示词。
- vLLM 单 GPU，最大上下文 4,096，最多 4 个序列，显存预算 50%，CUDA Graph 捕获大小 `[1,2,4]`；前缀缓存关闭。
- 每侧排除一个预热请求；3 个问题各测 2 次，再对同一提示词固定生成 128 token 测 3 次。
- 单次完整 RAG 计时包含检索、精排、压缩、生成；vLLM 包含本机 HTTP/SSE 开销，但不包含 Gradio 页面、SSH 隧道和浏览器开销。
- 同时保留原前端服务但测试串行执行；没有做多用户并发压测。

这是可实际部署的两种软件栈对比。PyTorch/CUDA 版本不同，不能把所有差异单独归因于 vLLM 中某一个优化。

## 结果

| 指标（中位数） | Transformers | vLLM |
|---|---:|---:|
| 固定 128 token，完整生成耗时 | 3.8378 秒 | 0.7605 秒 |
| 固定 128 token，输出 token/总耗时 | 约 33.35 token/s | 约 168.31 token/s |
| 固定 128 token，首 token/首文本分片 | 0.0331 秒 | 0.0397 秒 |
| 6 次自然回答，生成耗时 | 3.9597 秒 | 0.6775 秒 |
| 6 次自然回答，完整 RAG 耗时 | 4.0349 秒 | 0.7520 秒 |

固定输出长度下，完整生成约快 **5.05 倍**，耗时约减少 **80.2%**。首 token/首文本分片没有改善；两侧测量点略有差异，vLLM 还包含 HTTP 开销。

自然回答的完整 RAG 约快 5.37 倍，但部分回答长度不同，不能仅使用这组数据判断后端生成速度；固定 128 token 的结果更适合比较。

| 问题 | Transformers RAG 两次耗时（秒） | vLLM RAG 两次耗时（秒） | 两侧自然输出 token |
|---|---|---|---|
| 高血压患者日常如何管理血压？ | 4.0188 / 4.0509 | 0.8584 / 0.8488 | 132 / 132 |
| 糖尿病患者日常如何监测血糖？ | 7.7209 / 10.1974 | 0.7516 / 0.7523 | 256 / 116 |
| 高血压患者用药需要注意什么？ | 3.2714 / 3.2699 | 0.7020 / 0.7053 | 106 / 107 |

vLLM 日志记录模型加载约 1.71 秒；引擎初始化（profiling、KV cache、预热）约 53.53 秒。该时间不计入常驻推理延迟，也不包含依赖安装时间。Flash Attention 和 CUDA Graph 均启用；未安装 FlashInfer，采样回退至 PyTorch 实现。本次正式对比采用贪心解码。

## 一致性和限制

- 6 对请求的查询、检索 chunk ID、prompt SHA-256 和输入 token 数逐项一致。
- 固定 128 token 的全部三对输出 SHA-256 一致：`e1b88f37920b76fe4ebb9e14dc6a37800e3b1788be0dc5b38d9a33f463d6e27f`。
- 自然回答仅高血压管理问题的两对文本哈希一致，其他问题不完全一致。不同执行栈的数值差异可能影响贪心序列，不能承诺全部文本逐字相同。
- 两侧 6 次请求均为 `answered`，每次 3 条结构化 Citation，检索证据一致。
- 两侧原始回答都没有主动输出行内编号；当前后处理按原有逻辑补上合法引用。因此“有 Citation”仅验证结构和可追溯性，不能证明逐事实引用正确。
- Transformers 糖尿病问题触及 256 token 上限；vLLM 对应回答未触顶。自然回答长度差异会放大该问题的总耗时差异。
- 没有进行医学正确性标注、完整 Faithfulness 评估或并发吞吐量评估。
- 当前前端仍保留原采样设置；此处的贪心对比结果不等于原页面采样参数下的延迟承诺。

## 复现

先在独立 vLLM 环境安装匹配版本：

```bash
pip install vllm==0.11.0 transformers==4.57.6 openai==1.109.1
```

在独立环境启动本机测试服务（检查点路径自行填写）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/checkpoint-1082 \
  --served-model-name qwen3-medical-benchmark \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.5 --no-enable-prefix-caching \
  --disable-log-requests --seed 0 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4]}'
```

设置真实模型与 RAG 相关环境变量，然后分别在项目环境执行：

```bash
python scripts/benchmark_backends.py --backend transformers --rounds 2 \
  --output output/benchmark_transformers.json
python scripts/benchmark_backends.py --backend vllm --rounds 2 \
  --api-url http://127.0.0.1:8001/v1 \
  --output output/benchmark_vllm.json
```

结果文件只包含查询、哈希、指标及证据 ID，不包含真实语料段落或生成回答；`output/` 已在 `.gitignore` 中。原始计时结果留在服务器项目的 `output/` 目录。
