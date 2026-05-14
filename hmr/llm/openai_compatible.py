from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing_extensions import override

from hmr.utils import retry
from hmr.llm.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenAICompatibleLLMClient(LLMClient):
    """Low-level OpenAI-compatible chat completions client.

    This class is intentionally small. It lives outside core retrieval logic so
    changing providers does not affect Reader construction or query execution.
    """

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int | None = None

    @retry(retries=5, delay=1)
    @override
    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 80000, json_require: bool = False) -> str:
        logger.debug("Calling remote LLM model=%s max_tokens=%s", self.model, max_tokens)
        payload = self._payload(prompt, temperature, max_tokens, json_require)
        request = self._request(payload)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        logger.debug(f"llm response with content: \n{content}")
        return content

    def _payload(self, prompt: str, temperature: float, max_tokens: int, json_require: bool) -> bytes:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_require:
            payload["response_format"] = {
                'type': 'json_object'
            }
        return json.dumps(payload).encode("utf-8")

    def _request(self, payload: bytes) -> urllib.request.Request:
        url = self.base_url.rstrip("/") + "/chat/completions"
        return urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
