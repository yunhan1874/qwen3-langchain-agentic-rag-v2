from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Any

from .showcase import (
    DEMO_NOTICE,
    build_demo_engine,
    parse_metadata_filter,
    result_payload,
)


DEMO_MODE = "演示模式（合成数据）"
REAL_MODE = "真实模式（Qwen + BGE + Chroma）"
SCORE_HEADERS = ["chunk_id", "dense", "bm25", "rrf", "rerank", "selected", "source"]


@lru_cache(maxsize=1)
def _demo_engine() -> Any:
    return build_demo_engine()


@lru_cache(maxsize=1)
def _real_rag() -> Any:
    from .agent import LangChainAgenticRAG
    from .cli import build_engine
    from .config import RAGConfig

    return LangChainAgenticRAG(build_engine(RAGConfig()))


def _score_rows(payload: dict[str, Any]) -> list[list[Any]]:
    return [[row.get(header, "") for header in SCORE_HEADERS] for row in payload["scores"]]


def handle_request(
    query: str,
    mode: str,
    metadata_filter: str,
) -> tuple[str, Any, str, str, str, list[list[Any]]]:
    """UI callback kept independent from Gradio so it can be smoke-tested offline."""
    try:
        filters = parse_metadata_filter(metadata_filter)
        is_demo = mode != REAL_MODE
        if is_demo:
            result = _demo_engine().run(query, filters)
        else:
            result = _real_rag().invoke(query, filters)
        payload = result_payload(result, demo_mode=is_demo)
        query_view = (
            f"**Original Query**\n\n{payload['original_query']}\n\n"
            f"**Rewritten Query**\n\n{payload['rewritten_query']}"
        )
        steps = "\n".join(
            f"{index}. `{step}`" for index, step in enumerate(payload["agent_steps"], 1)
        )
        status = (
            f"**状态：** `{payload['status']}`  "
            f"**置信度：** `{payload['confidence']:.4f}`  "
            f"**模式：** `{'DEMO / SYNTHETIC' if is_demo else 'REAL'}`"
        )
        return (
            payload["answer_markdown"],
            payload["citations"],
            query_view,
            steps,
            status,
            _score_rows(payload),
        )
    except Exception as exc:
        mode_hint = (
            "真实模式初始化或执行失败。请检查 .env、模型、语料和索引配置。"
            if mode == REAL_MODE
            else "演示请求执行失败。"
        )
        return (
            f"### 无法完成请求\n\n{mode_hint}\n\n`{type(exc).__name__}: {exc}`",
            [],
            "",
            "",
            f"**状态：** `error`  **模式：** `{mode}`",
            [],
        )


def build_ui(demo_only: bool = False, default_real: bool = False) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Web UI 需要可选依赖：pip install -e '.[ui]'") from exc

    modes = [DEMO_MODE] if demo_only else [DEMO_MODE, REAL_MODE]
    default_mode = REAL_MODE if default_real and not demo_only else DEMO_MODE
    with gr.Blocks(
        title="Qwen3 LangChain Agentic RAG v2",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Qwen3 LangChain Agentic RAG v2\n"
            "可观察 Query Rewrite、Metadata Filter、Hybrid Retrieval、Reranker、"
            "Agent 决策、Context Compression 与 Citation 的作品集演示。"
        )
        gr.Markdown(f"> ⚠️ **演示模式说明：{DEMO_NOTICE}**")
        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="问题",
                    lines=3,
                    placeholder="例如：请问血压记录可以包含哪些内容？",
                )
                mode = gr.Radio(modes, value=default_mode, label="运行模式")
                metadata_filter = gr.Textbox(
                    label="Metadata Filter（JSON）",
                    value="{}",
                    placeholder='{"department": "心内科"}',
                )
                submit = gr.Button("运行 Agent", variant="primary")
            with gr.Column(scale=3):
                status = gr.Markdown("**状态：** 等待输入")
                answer = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("结构化 Citation"):
                citations = gr.JSON(label="Citations")
            with gr.Tab("Query Rewrite"):
                query_view = gr.Markdown()
            with gr.Tab("Agent 步骤"):
                steps = gr.Markdown()
            with gr.Tab("检索与重排分数"):
                scores = gr.Dataframe(
                    headers=SCORE_HEADERS,
                    datatype=["str", "number", "number", "number", "number", "bool", "str"],
                    interactive=False,
                    wrap=True,
                )

        gr.Examples(
            examples=[
                ["高血压患者如何管理血压？", default_mode, "{}"],
                ["糖尿病患者如何进行日常血糖管理？", default_mode, "{}"],
                ["出现呼吸道症状时应记录哪些信息？", default_mode, "{}"],
                ["高血压患者用药需要注意什么？", default_mode, "{}"],
                ["火箭发动机的推力是多少？", default_mode, "{}"],
            ],
            inputs=[question, mode, metadata_filter],
            label="示例问题",
        )
        submit.click(
            handle_request,
            inputs=[question, mode, metadata_filter],
            outputs=[answer, citations, query_view, steps, status, scores],
        )
        question.submit(
            handle_request,
            inputs=[question, mode, metadata_filter],
            outputs=[answer, citations, query_view, steps, status, scores],
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 Agentic RAG v2 Web UI")
    parser.add_argument("--demo-only", action="store_true", help="隐藏真实模型模式")
    parser.add_argument("--default-real", action="store_true", help="默认选择真实模型模式")
    parser.add_argument(
        "--host",
        default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
    parser.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_ui(demo_only=args.demo_only, default_real=args.default_real).queue(
        default_concurrency_limit=1
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
