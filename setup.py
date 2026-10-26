"""Compatibility metadata for the server's legacy pip/setuptools toolchain."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    name="qwen3-langchain-agentic-rag-v2",
    version="2.0.0",
    description="LangChain-based Qwen3 medical Agentic RAG with semantic chunking",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(include=("rag_v2", "rag_v2.*")),
    install_requires=[
        "accelerate>=1.0",
        "chromadb>=1.0,<2",
        "langchain-core>=0.3,<2",
        "langchain-huggingface>=0.1,<2",
        "sentence-transformers>=3.4,<4",
        "transformers>=4.51,<5",
        "typing-extensions>=4.12",
    ],
    extras_require={
        "dev": ["ruff>=0.9", "pytest>=8"],
        "ui": ["gradio>=5,<7"],
        "train": [
            "datasets>=3,<5",
            "modelscope>=1.20,<2",
            "pandas>=2,<3",
            "swanlab>=0.5,<1",
        ],
    },
    entry_points={
        "console_scripts": [
            "qwen3-agentic-rag=rag_v2.cli:main",
            "qwen3-agentic-rag-web=rag_v2.webapp:main",
        ]
    },
)
