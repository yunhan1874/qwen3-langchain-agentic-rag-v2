"""新版入口；RAG 实现在 rag_v2 包中，原训练与数据脚本保持不变。"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from rag_v2.cli import main


if __name__ == "__main__":
    main()
