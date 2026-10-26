from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class VLLMCompletionLLM:
    """Raw-text completion adapter; keeps the existing trained-model prompt format."""

    base_url: str
    model: str
    max_new_tokens: int = 256
    timeout_seconds: float = 120
    api_key: str = field(default="", repr=False)

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"vLLM 请求失败（HTTP {exc.code}），请检查推理服务。") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("无法连接 vLLM 推理服务，或请求超时。") from exc
        except ValueError as exc:
            raise RuntimeError("vLLM 返回了无效 JSON。") from exc
        if not isinstance(data, dict):
            raise RuntimeError("vLLM 返回的数据必须为 JSON 对象。")
        return data

    def check_ready(self) -> None:
        data = replace(self, timeout_seconds=min(self.timeout_seconds, 10))._request("/models")
        ids = {item.get("id") for item in data.get("data", []) if isinstance(item, dict)}
        if self.model not in ids:
            raise RuntimeError(f"vLLM 服务没有加载配置的模型：{self.model}")

    def bind(self, *, pipeline_kwargs: dict[str, Any]) -> VLLMCompletionLLM:
        limit = pipeline_kwargs["max_new_tokens"]
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("max_new_tokens 必须为正整数")
        return replace(self, max_new_tokens=limit)

    def invoke(self, prompt: str) -> str:
        data = self._request("/completions", {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            # Match the original pipeline's effective sampling settings.
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 20,
            "stream": False,
        })
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("vLLM 返回了无效的 completion。")
        text = choices[0].get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("vLLM 没有返回可用回答。")
        return text
