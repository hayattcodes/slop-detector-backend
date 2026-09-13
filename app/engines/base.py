"""
Every detection engine (perplexity/burstiness, website fingerprint, and
future engines we add — CNN classifier, LLM-judge, the heavy ensemble
running on the Cornell server, etc.) implements this same shape.

Keeping a consistent interface means:
- adding a new engine later = write one function, register it, done
- the scoring step + frontend don't need to know engine internals
- we can run engines in parallel and stream results (SlopTotal-style)
  once we wire up SSE, without changing this contract
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EngineOutput:
    engine_id: str
    engine_name: str
    category: Literal["content", "website"]
    score: float  # 0-100, higher = more likely AI-generated / AI-built
    confidence: Literal["low", "medium", "high"]
    summary: str
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "category": self.category,
            "score": round(self.score, 1),
            "confidence": self.confidence,
            "summary": self.summary,
            "details": self.details,
        }


class Engine(ABC):
    engine_id: str
    engine_name: str
    category: Literal["content", "website"]
    # Reliability weight used when combining engines into the overall score.
    # Start conservative (lightweight, unvalidated engines get less say);
    # will be re-tuned once we have a labeled dataset to calibrate against.
    weight: float = 1.0

    @abstractmethod
    def applies_to(self, input_type: str) -> bool:
        """Whether this engine should run for a given input_type
        ("url" | "text" | "file" | "image")."""
        raise NotImplementedError

    @abstractmethod
    def run(self, *, text: str = "", html: str = "", url: str = "", raw_html_before_js: str = "") -> EngineOutput:
        raise NotImplementedError
