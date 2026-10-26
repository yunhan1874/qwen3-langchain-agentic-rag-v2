import json
import os

# ====== 配置 ======
train_jsonl_path = os.getenv("TRAIN_FORMAT_PATH", "./data/train_format.jsonl")
docs_output_path = os.getenv("RAG_DOCS_PATH", "./data/docs.txt")

if not os.path.exists(train_jsonl_path):
    raise FileNotFoundError(f"训练集文件不存在：{train_jsonl_path}")

docs = []

with open(train_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        input_text = data.get("input", "")
        output_text = data.get("output", "")
        # 将思考步骤与答案拼接成文档
        doc_text = f"问题: {input_text}\n答案: {output_text}"
        docs.append(doc_text.strip())

# 保存到 docs.txt
with open(docs_output_path, "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(doc + "\n")

print(f"已生成 RAG 文档库: {docs_output_path}, 共 {len(docs)} 条文档")
