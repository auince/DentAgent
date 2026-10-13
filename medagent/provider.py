"""Bounded local model calls. Endpoints and credentials are explicit configuration."""
import base64
import contextvars
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .common import ROOT, read_json

CALL_METADATA=contextvars.ContextVar('local_model_call_metadata',default=None)


class ModelUnavailable(RuntimeError):
    pass


class LocalModelClient:
    def __init__(self, config=None):
        self.config = config or read_json(ROOT / "configs/models.json")

    def complete(self, role, system, text, images=(), response_format=None):
        CALL_METADATA.set(None)
        cfg = self.config[role]
        endpoint = os.environ.get(cfg.get("endpoint_env", ""), cfg["endpoint"])
        if urlsplit(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ModelUnavailable("模型端点必须为本机回环地址；通过 SSH 转发连接远端服务")
        content = [{"type": "text", "text": text}]
        for image in images:
            if not isinstance(image, str) or not image.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
                raise ValueError("仅接受 PNG/JPEG/WebP base64 图像")
            if len(image) > 16_000_000:
                raise ValueError("图像超过大小限制")
            base64.b64decode(image.split(",", 1)[1], validate=True)
            content.append({"type": "image_url", "image_url": {"url": image}})
        body = {"model": cfg["model"], "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content if images else text}],
            "temperature": cfg.get("temperature", 0), "max_tokens": cfg.get("max_tokens", 4096),
            "chat_template_kwargs": {"enable_thinking": False}}
        if response_format is not None:body['response_format']=response_format
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(cfg.get("api_key_env", ""))
        if key:
            headers["Authorization"] = "Bearer " + key
        retries = cfg.get('retries', 1)
        for attempt in range(retries + 1):
            req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)
            try:
                with urllib.request.urlopen(req, timeout=cfg.get("timeout", 180)) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                CALL_METADATA.set({'model':result.get('model',cfg['model']),
                    'usage':result.get('usage'),'finish_reason':choice.get('finish_reason')})
                if choice.get("finish_reason") == "length":
                    raise ModelUnavailable(role + ": output_truncated")
                return choice["message"]["content"]
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt >= retries:
                    raise ModelUnavailable(f"{role}: HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt >= retries:
                    raise ModelUnavailable(f"{role}: local endpoint unavailable") from exc
            time.sleep(1)
        raise ModelUnavailable(role)
