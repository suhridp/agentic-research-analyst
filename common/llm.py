"""Provider-agnostic LLM access with the guarantees an experiment needs.

Every call goes through `BaseLLM.complete`, which adds:
  * a disk cache keyed by (provider, model, temperature, system, prompt, max_tokens), so reruns are
    free and results are reproducible even if the provider changes behaviour later;
  * retries with exponential backoff (honouring Retry-After) on 429/5xx/timeouts;
  * token and cost accounting, plus a hard spend cap that stops an experiment before it overspends;
  * an optional tracer hook so every call shows up in the run trace.

Providers are plain HTTP (stdlib only) so the package has no SDK dependency:
  * AnthropicLLM      - Messages API
  * OpenAICompatLLM   - any /v1/chat/completions endpoint (OpenAI, Ollama, vLLM, Together, ...),
                        which is how an open-weight model is plugged in
  * ScriptedLLM       - deterministic fake for tests; lets the LLM code paths run offline
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import tracing

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMError(RuntimeError):
    """Non-retryable provider error (bad request, auth, exhausted retries)."""


class BudgetExceeded(RuntimeError):
    """Raised before a call that would take the run past its spend cap."""


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    retries: int = 0

    def snapshot(self) -> Usage:
        return Usage(**asdict(self))

    def minus(self, other: Usage) -> Usage:
        return Usage(**{k: getattr(self, k) - getattr(other, k) for k in asdict(self)})


@dataclass
class Price:
    """USD per million tokens. Prices change: keep them in configs/prices.yaml, not in code."""
    input: float
    output: float


@dataclass
class Budget:
    max_cost_usd: float | None = None
    max_tokens: int | None = None   # input + output; works even when no price is configured


class _Transient(Exception):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def _hash(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class BaseLLM:
    provider = "base"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        cache_dir: str | Path | None = ".llm_cache",
        price: Price | None = None,
        budget: Budget | None = None,
        max_retries: int = 5,
        timeout_s: float = 120.0,
        backoff_base_s: float = 1.0,
        tracer: Any = None,
    ):
        self.model = model
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.price = price
        self.budget = budget or Budget()
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.backoff_base_s = backoff_base_s
        self.tracer = tracer
        self.usage = Usage()
        self._lock = threading.Lock()

    # -- identity -------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "temperature": self.temperature}

    # -- public API -----------------------------------------------------------------------
    def complete(self, system: str, prompt: str, max_tokens: int = 1024, *, tag: str = "") -> str:
        key = _hash(self.provider, self.model, self.temperature, system, prompt, max_tokens)
        cached = self._cache_get(key)
        if cached is not None:
            with self._lock:
                self.usage.cache_hits += 1
            _record(True)
            self._trace(tag, system, prompt, cached, 0, 0, 0.0, cached=True)
            return cached

        self._check_budget()
        t0 = time.perf_counter()
        attempt = 0
        while True:
            try:
                text, tin, tout = self._call(system, prompt, max_tokens)
                break
            except _Transient as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise LLMError(f"{self.name}: gave up after {self.max_retries} retries: {e}") from e
                with self._lock:
                    self.usage.retries += 1
                delay = e.retry_after or self.backoff_base_s * (2 ** (attempt - 1))
                time.sleep(min(delay, 60.0) * (0.75 + random.random() / 2))
        dt = time.perf_counter() - t0
        cost = (tin * self.price.input + tout * self.price.output) / 1e6 if self.price else 0.0
        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += tin
            self.usage.output_tokens += tout
            self.usage.latency_s += dt
            self.usage.cost_usd += cost
        _record(False, tin, tout, cost, dt)
        self._cache_put(key, text, tin, tout)
        self._trace(tag, system, prompt, text, tin, tout, dt, cached=False)
        return text

    def complete_json(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        *,
        validate: Callable[[Any], bool] | None = None,
        tag: str = "",
    ) -> Any | None:
        """Ask for JSON; on unparseable or invalid output, make exactly one repair attempt.
        Returns None if both attempts fail, so callers must handle the failure explicitly."""
        text = self.complete(system, prompt, max_tokens, tag=tag)
        out = parse_json(text)
        if out is not None and (validate is None or validate(out)):
            return out
        repair = (f"{prompt}\n\nYour previous reply was not valid for the required JSON format:\n"
                  f"{text[:1500]}\n\nReply again with ONLY the JSON, no prose and no code fences.")
        out = parse_json(self.complete(system, repair, max_tokens, tag=f"{tag}:repair"))
        if out is not None and (validate is None or validate(out)):
            return out
        return None

    # -- provider hook --------------------------------------------------------------------
    def _call(self, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        raise NotImplementedError

    # -- helpers --------------------------------------------------------------------------
    def _check_budget(self) -> None:
        b, u = self.budget, self.usage
        if b.max_cost_usd is not None and u.cost_usd >= b.max_cost_usd:
            raise BudgetExceeded(f"{self.name}: spend cap ${b.max_cost_usd:.2f} reached (${u.cost_usd:.4f})")
        if b.max_tokens is not None and u.input_tokens + u.output_tokens >= b.max_tokens:
            raise BudgetExceeded(f"{self.name}: token cap {b.max_tokens} reached")

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def _cache_get(self, key: str) -> str | None:
        p = self._cache_path(key)
        if p and p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))["text"]
            except (json.JSONDecodeError, KeyError, OSError):
                return None  # corrupt entry: treat as a miss and overwrite
        return None

    def _cache_put(self, key: str, text: str, tin: int, tout: int) -> None:
        p = self._cache_path(key)
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"model": self.name, "text": text, "input_tokens": tin,
                                   "output_tokens": tout}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)  # atomic, so parallel workers never read half-written entries

    def _trace(self, tag, system, prompt, text, tin, tout, dt, cached):
        tracer = self.tracer or tracing.current()
        if tracer is not None:
            tracer.event("llm_call", tag=tag, model=self.name, cached=cached,
                              input_tokens=tin, output_tokens=tout, latency_s=round(dt, 4),
                              system=system[:300], prompt=prompt[:2000], output=text[:2000])

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in RETRYABLE_STATUS:
                ra = e.headers.get("retry-after") if e.headers else None
                raise _Transient(f"HTTP {e.code}: {detail}", float(ra) if ra and ra.isdigit() else None) from e
            raise LLMError(f"{self.name}: HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise _Transient(f"network: {e}") from e


class AnthropicLLM(BaseLLM):
    provider = "anthropic"

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None, **kw):
        super().__init__(model, **kw)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        # Deliberately not ANTHROPIC_BASE_URL, so a tool-specific proxy in the shell can't hijack runs.
        self.base_url = (base_url or os.getenv("ARA_ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")

    def _call(self, system, prompt, max_tokens):
        data = self._post(f"{self.base_url}/v1/messages",
                          {"content-type": "application/json", "x-api-key": self.api_key,
                           "anthropic-version": "2023-06-01"},
                          {"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature,
                           "system": system, "messages": [{"role": "user", "content": prompt}]})
        u = data.get("usage", {})
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return text, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))


class OpenAICompatLLM(BaseLLM):
    """Any OpenAI-style chat endpoint. For a local open-weight model with Ollama:
    provider: openai_compat, base_url: http://localhost:11434/v1, model: qwen2.5:7b-instruct"""
    provider = "openai_compat"

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None, **kw):
        super().__init__(model, **kw)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "not-needed-for-local"

    def _call(self, system, prompt, max_tokens):
        data = self._post(f"{self.base_url}/chat/completions",
                          {"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
                          {"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}]})
        u = data.get("usage") or {}
        text = data["choices"][0]["message"].get("content") or ""
        return text, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))


class ScriptedLLM(BaseLLM):
    """Deterministic stand-in for a real model: `responder(system, prompt) -> text`.
    Token counts are approximated as words so cost/budget logic is exercised in tests."""
    provider = "scripted"

    def __init__(self, responder: Callable[[str, str], str], model: str = "scripted", **kw):
        kw.setdefault("cache_dir", None)
        super().__init__(model, **kw)
        self.responder = responder

    def _call(self, system, prompt, max_tokens):
        text = self.responder(system, prompt)
        return text, len((system + prompt).split()), len(text.split())


def parse_json(text: str) -> Any | None:
    """Tolerant JSON extraction: strips code fences and falls back to the first {...} or [...] block."""
    if not text:
        return None
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_, close in (("[", "]"), ("{", "}")):
        i, j = t.find(open_), t.rfind(close)
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def build_llm(cfg: dict | None, *, prices: dict[str, dict] | None = None, tracer: Any = None) -> BaseLLM | None:
    """Factory used by configs. `cfg` is None or {"provider": "none"} for offline heuristic mode."""
    if not cfg or cfg.get("provider", "none") in ("none", "offline", None):
        return None
    cfg = dict(cfg)
    provider = cfg.pop("provider")
    model = cfg.pop("model")
    p = (prices or {}).get(model)
    kw: dict[str, Any] = {
        "temperature": cfg.pop("temperature", 0.0),
        "cache_dir": cfg.pop("cache_dir", ".llm_cache"),
        "price": Price(**p) if p else None,
        "budget": Budget(cfg.pop("max_cost_usd", None), cfg.pop("max_tokens", None)),
        "max_retries": cfg.pop("max_retries", 5),
        "timeout_s": cfg.pop("timeout_s", 120.0),
        "tracer": tracer,
    }
    if provider == "anthropic":
        return AnthropicLLM(model, base_url=cfg.pop("base_url", None), **kw)
    if provider == "openai_compat":
        return OpenAICompatLLM(model, base_url=cfg.pop("base_url", None), **kw)
    raise ValueError(f"unknown LLM provider: {provider}")


class UsageMeter:
    """Attributes LLM usage to a unit of work (one question, one system) on the current thread.
    Thread-local, so per-question cost stays correct when questions run in parallel on a shared client.

        with UsageMeter() as m:
            run_pipeline(...)
        m.usage.cost_usd
    """

    def __init__(self):
        self.usage = Usage()

    def __enter__(self):
        _meters().append(self)
        return self

    def __exit__(self, *exc):
        _meters().remove(self)
        return False


_local = threading.local()


def _meters() -> list[UsageMeter]:
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def _record(cache_hit: bool, tin: int = 0, tout: int = 0, cost: float = 0.0, dt: float = 0.0) -> None:
    for m in _meters():
        u = m.usage
        if cache_hit:
            u.cache_hits += 1
        else:
            u.calls += 1
            u.input_tokens += tin
            u.output_tokens += tout
            u.cost_usd += cost
            u.latency_s += dt
