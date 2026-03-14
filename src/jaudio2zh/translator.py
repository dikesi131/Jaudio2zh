from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from typing import Any
from urllib import parse
from urllib import error, request

from tqdm import tqdm


class SakuraTranslator:
    def __init__(
        self,
        model_name_or_path: str,
        api_base: str,
        api_key: str,
        request_timeout: int,
        parallel_requests: int,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key.strip()
        self.request_timeout = max(10, int(request_timeout))
        self.parallel_requests = max(1, int(parallel_requests))
        self.disable_proxy = self._should_disable_proxy(self.api_base)
        self.model_id = self._resolve_model_id(model_name_or_path)
        self.logger.info(
            "Using Sakura API translator model=%s base=%s",
            self.model_id,
            self.api_base,
        )

    def translate_texts(self, ja_texts: list[str]) -> list[str]:
        if not ja_texts:
            return []

        self.logger.info("Translate sentence-by-sentence mode enabled")
        total = len(ja_texts)

        if self.parallel_requests == 1:
            outputs: list[str] = []
            iterator = tqdm(ja_texts, desc="Translating")
            for idx, text in enumerate(iterator, start=1):
                outputs.append(self._translate_one(text, idx=idx, total=total))
            return outputs

        outputs = [""] * len(ja_texts)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.parallel_requests
        ) as executor:
            future_to_idx = {
                executor.submit(
                    self._translate_one,
                    text,
                    idx=idx + 1,
                    total=total,
                ): idx
                for idx, text in enumerate(ja_texts)
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(ja_texts),
                desc="Translating",
            ):
                idx = future_to_idx[future]
                outputs[idx] = future.result()
        return outputs

    def _resolve_model_id(self, model_name_or_path: str) -> str:
        if model_name_or_path and model_name_or_path.lower() != "auto":
            return model_name_or_path

        models_url = self._join_url("/v1/models")
        req = request.Request(models_url, method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with self._open_url(req) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            data = payload.get("data", [])
            if data and isinstance(data, list) and data[0].get("id"):
                return str(data[0]["id"])
        except Exception as exc:
            self.logger.warning("Cannot auto-detect model id: %s", exc)

        # Common alias used with local OpenAI-compatible servers.
        return "sakura"

    def _translate_one(self, text: str, idx: int, total: int) -> str:
        japanese = text
        user_prompt, system_prompt = self._build_prompts(japanese)
        print(f"\n[{idx}/{total}] Translating segment", flush=True)
        print(f"[{idx}/{total}] Source: {japanese}", flush=True)
        print(f"[{idx}/{total}] system_prompt: {system_prompt}", flush=True)
        print(f"[{idx}/{total}] user_prompt: {user_prompt}", flush=True)

        payload = {
            "model": self.model_id,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        content = self._chat_completion_content(payload)
        if content:
            print(f"[{idx}/{total}] Result: {content}", flush=True)
            return content
        print(
            f"[{idx}/{total}] Translation failed, keep source text",
            flush=True,
        )
        return text

    def _chat_completion_content(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        url = self._join_url("/v1/chat/completions")

        for attempt in range(3):
            req = request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")

            try:
                with self._open_url(req) as resp:
                    raw = resp.read().decode("utf-8")
                data: dict[str, Any] = json.loads(raw)
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = str(message.get("content", "")).strip()
                    if content:
                        return content
                raise RuntimeError("empty completion content")
            except (
                error.URLError,
                error.HTTPError,
                TimeoutError,
                ValueError,
            ) as exc:
                if attempt >= 2:
                    self.logger.error("Translation failed: %s", exc)
                    return ""
                time.sleep(0.5 * (attempt + 1))
            except Exception as exc:
                self.logger.error("Translation failed: %s", exc)
                return ""

        return ""

    def _build_prompts(self, japanese: str) -> tuple[str, str]:
        user_prompt = "将该日文文本翻译成中文: " + japanese
        system_prompt = (
            "\n你是一个轻小说翻译模型, 可以流畅通顺地以日本轻小说的风格, 将日文翻译成简体中文, 不擅自添加原文中没有的代词。"
        )
        return user_prompt, system_prompt

    def _join_url(self, path: str) -> str:
        if self.api_base.endswith("/v1") and path.startswith("/v1/"):
            return self.api_base + path[3:]
        return self.api_base + path

    def _open_url(self, req: request.Request):
        if self.disable_proxy:
            opener = request.build_opener(request.ProxyHandler({}))
            return opener.open(req, timeout=self.request_timeout)
        return request.urlopen(req, timeout=self.request_timeout)

    def _should_disable_proxy(self, api_base: str) -> bool:
        host = (parse.urlparse(api_base).hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1"}
