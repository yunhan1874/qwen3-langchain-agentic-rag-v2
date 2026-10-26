from __future__ import annotations

import importlib
import unittest

from rag_v2.showcase import parse_metadata_filter, run_demo
from rag_v2.webapp import DEMO_MODE, handle_request


class ShowcaseTests(unittest.TestCase):
    def test_demo_exposes_complete_agent_observability(self) -> None:
        payload = run_demo("请问血压记录可以包含哪些内容？")
        self.assertEqual("answered", payload["status"])
        self.assertTrue(payload["citations"])
        self.assertTrue(payload["scores"])
        self.assertTrue(any(row["selected"] for row in payload["scores"]))
        self.assertIn("演示模式", payload["answer_markdown"])
        self.assertTrue(any("retrieve_attempt" in step for step in payload["agent_steps"]))

    def test_demo_metadata_filter_applies_to_both_retrieval_routes(self) -> None:
        payload = run_demo("随访时可以记录什么？", '{"department":"内分泌科"}')
        self.assertTrue(payload["scores"])
        self.assertEqual(
            {"demo-endo-001"},
            {row["chunk_id"] for row in payload["scores"]},
        )

    def test_demo_abstains_for_out_of_domain_question(self) -> None:
        payload = run_demo("火箭发动机的推力是多少？")
        self.assertEqual("abstained", payload["status"])
        self.assertEqual([], payload["citations"])

    def test_filter_rejects_non_object_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON 对象"):
            parse_metadata_filter("[1, 2]")

    def test_ui_callback_smoke_without_importing_gradio(self) -> None:
        importlib.import_module("rag_v2.webapp")
        answer, citations, query_view, steps, status, scores = handle_request(
            "这个系统怎样融合检索并提供引用？",
            DEMO_MODE,
            '{"department":"工程"}',
        )
        self.assertIn("DEMO / SYNTHETIC", status)
        self.assertIn("Original Query", query_view)
        self.assertIn("retrieve_attempt", steps)
        self.assertTrue(citations)
        self.assertTrue(scores)
        self.assertIn("演示模式", answer)


if __name__ == "__main__":
    unittest.main()
