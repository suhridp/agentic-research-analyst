from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..corpus import Passage


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...

    def describe(self) -> dict: ...
