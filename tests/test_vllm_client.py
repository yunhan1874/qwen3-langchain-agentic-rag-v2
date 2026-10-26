from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_v2.cli import _build_llm
from rag_v2.config import RAGConfig
from rag_v2.showcase import build_demo_engine
from rag_v2.vllm_client import VLLMCompletionLLM
from rag_v2.webapp import REAL_MODE, handle_request


class FakeVLLMHandler(BaseHTTPRequestHandler):
    calls = []
    status = 200
    response = {"choices": [{"text": "证据回答。[1]"}]}

    def log_message(self, *args):
        pass

    def send_json(self, data, status=200):
        encoded = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        self.send_json({"data": [{"id": "qwen3-medical"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, payload))
        self.send_json(self.response, self.status)


class VLLMClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        FakeVLLMHandler.calls.clear()
        FakeVLLMHandler.status = 200
        FakeVLLMHandler.response = {"choices": [{"text": "证据回答。[1]"}]}

    def test_raw_prompt_and_independent_token_budgets_reach_http_server(self):
        llm = VLLMCompletionLLM(self.url, "qwen3-medical")
        llm.check_ready()
        self.assertEqual("证据回答。[1]", llm.invoke("原始证据\n问题"))
        llm.bind(pipeline_kwargs={"max_new_tokens": 96}).invoke("改写提示词")
        calls = FakeVLLMHandler.calls
        self.assertEqual([256, 96], [body["max_tokens"] for _, body in calls])
        self.assertEqual("原始证据\n问题", calls[0][1]["prompt"])
        self.assertEqual("/v1/completions", calls[0][0])
        self.assertEqual(0.1, calls[0][1]["temperature"])
        self.assertEqual(256, llm.max_new_tokens)

    def test_readiness_rejects_wrong_model(self):
        with self.assertRaisesRegex(RuntimeError, "没有加载"):
            VLLMCompletionLLM(self.url, "wrong-model").check_ready()

    def test_server_failure_is_visible_and_does_not_load_transformers(self):
        FakeVLLMHandler.status = 503
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            VLLMCompletionLLM(self.url, "qwen3-medical").invoke("问题")
        self.assertEqual(1, len(FakeVLLMHandler.calls))

    def test_malformed_or_empty_completion_is_rejected(self):
        for data in ({}, {"choices": []}, {"choices": [{"text": ""}]}):
            with self.subTest(data=data):
                FakeVLLMHandler.response = data
                with self.assertRaises(RuntimeError):
                    VLLMCompletionLLM(self.url, "qwen3-medical").invoke("问题")

    def test_vllm_factory_does_not_import_local_model_dependencies(self):
        config = replace(RAGConfig(), llm_backend="vllm", vllm_base_url=self.url)
        with patch.dict("sys.modules", {"transformers": None, "langchain_huggingface": None}):
            llm = _build_llm(config)
        self.assertIsInstance(llm, VLLMCompletionLLM)

    def test_bad_backend_configuration_is_rejected(self):
        for config in (
            replace(RAGConfig(), llm_backend="unknown"),
            replace(RAGConfig(), llm_backend="vllm", vllm_base_url="file:///tmp/model"),
            replace(RAGConfig(), llm_backend="vllm", vllm_model=""),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_real_ui_reports_selected_backend(self):
        engine = build_demo_engine()
        fake_rag = SimpleNamespace(
            engine=SimpleNamespace(config=replace(RAGConfig(), llm_backend="vllm")),
            invoke=engine.run,
        )
        with patch("rag_v2.webapp._real_rag", return_value=fake_rag):
            response = handle_request("请问血压记录可以包含哪些内容？", REAL_MODE, "{}")
        self.assertIn("`vLLM`", response[4])
        self.assertIn("`answered`", response[4])
        self.assertTrue(response[1])


if __name__ == "__main__":
    unittest.main()
