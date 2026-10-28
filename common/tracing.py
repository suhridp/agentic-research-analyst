"""Structured run traces: every pipeline step and LLM call becomes one JSON event.

A Tracer is activated per unit of work (one question) on the current thread, so components deep in
the call stack - including the shared LLM client - log to the right trace without passing it around:

    tr = Tracer(run_id="q17/agentic_full")
    with tr.activate():
        with span("retrieve", query=q) as s:
            s["n"] = len(hits)
    tr.dump("results/.../traces/q17.agentic_full.jsonl")
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_local = threading.local()


class Tracer:
    def __init__(self, run_id: str = ""):
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"t": round(time.perf_counter() - self._t0, 4), "kind": kind, **_jsonable(data)})

    @contextmanager
    def activate(self) -> Iterator[Tracer]:
        stack = _stack()
        stack.append(self)
        try:
            yield self
        finally:
            stack.remove(self)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps({"run_id": self.run_id, **e}, ensure_ascii=False) + "\n")


def _stack() -> list[Tracer]:
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def current() -> Tracer | None:
    s = _stack()
    return s[-1] if s else None


def event(kind: str, **data: Any) -> None:
    """Log to the active tracer, if any. Safe to call when tracing is off."""
    t = current()
    if t is not None:
        t.event(kind, **data)


@contextmanager
def span(name: str, **inputs: Any) -> Iterator[dict[str, Any]]:
    """Time a step. Put outputs in the yielded dict; they are logged when the step ends."""
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        event("step", step=name, latency_s=round(time.perf_counter() - t0, 4), inputs=inputs, outputs=out)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "to_dict"):
        return _jsonable(x.to_dict())
    return str(x)
