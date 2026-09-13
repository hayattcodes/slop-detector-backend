"""
Perplexity + Burstiness engine.

Adapted from:
- Repo 1 (AkashKobal/Generative-AI-Detection): sliding-window GPT-2 perplexity,
  per-sentence burstiness, 3-tier threshold classification.
- Repo 6 (PrishitaK/AI-generated-Content-Detection): the simpler
  perplexity + word-repetition-burstiness combination as a sanity check.

Why this engine first: it needs no training data (zero-shot), the model
(GPT-2 base, ~500MB) is small enough to run comfortably on a laptop CPU,
and it's a reasonable always-on baseline signal while heavier engines
are added later.

Important caveat we're upfront about: these thresholds are the ones used
in the source repos, not yet validated against our own labeled dataset.
Once Pranav's dataset is available, this is the first engine to recalibrate.
"""
import re
from functools import lru_cache

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from app.engines.base import Engine, EngineOutput

_MODEL_ID = "distilgpt2"
_STRIDE = 512
_MAX_LENGTH = 1024


@lru_cache(maxsize=1)
def _load_model():
    """Loaded once per process (not per-request) — this is the expensive part."""
    tokenizer = GPT2TokenizerFast.from_pretrained(_MODEL_ID)
    model = GPT2LMHeadModel.from_pretrained(_MODEL_ID)
    model.eval()
    return tokenizer, model


def _sentence_split(text: str) -> list[str]:
    # Same lookbehind-regex approach as Repo 1: split on sentence-ending
    # punctuation followed by whitespace, keep the punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _perplexity(text: str, tokenizer, model) -> float:
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    seq_len = encodings.input_ids.size(1)
    if seq_len == 0:
        return 0.0

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, _STRIDE):
        end_loc = min(begin_loc + _MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs.loss * trg_len
        nlls.append(nll)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return float(ppl.item())


def _word_repetition_burstiness(text: str) -> float:
    """Repo 6's simpler burstiness metric: ratio of repeated-word count to
    unique-word count. Lower = more AI-like (even, predictable vocabulary)."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    repeated = sum(1 for c in freq.values() if c > 1)
    return repeated / len(freq)


class PerplexityBurstinessEngine(Engine):
    engine_id = "perplexity_burstiness"
    engine_name = "Perplexity & Burstiness"
    category = "content"
    weight = 0.25  # GPT-2 unreliable on modern LLM text — supporting signal only

    def applies_to(self, input_type: str) -> bool:
        return input_type in ("text", "url", "file")

    def run(self, *, text: str = "", html: str = "", url: str = "", raw_html_before_js: str = "") -> EngineOutput:
        text = text.strip()
        if len(text) < 100:
            return EngineOutput(
                engine_id=self.engine_id,
                engine_name=self.engine_name,
                category=self.category,
                score=0.0,
                confidence="low",
                summary="Text is too short (<100 characters) for a reliable perplexity reading.",
                details={"reason": "insufficient_length"},
            )

        tokenizer, model = _load_model()

        sentences = _sentence_split(text)
        full_ppl = _perplexity(text, tokenizer, model)
        sentence_ppls = [_perplexity(s, tokenizer, model) for s in sentences[:40]]  # cap for speed
        burstiness_ppl = max(sentence_ppls) if sentence_ppls else full_ppl
        repetition_burstiness = _word_repetition_burstiness(text)

        # Threshold logic adapted from Repo 1 (sentence-level perplexity
        # tiers) blended with Repo 6's repetition-burstiness as a secondary
        # signal, rather than a strict AND gate — a gate is too brittle
        # (Repo 6 itself notes it has no gradient/confidence scoring).
        if burstiness_ppl < 60:
            base_score = 85
        elif burstiness_ppl < 80:
            base_score = 60
        else:
            base_score = 25

        # Nudge the score using repetition-burstiness: very low repetition
        # (< 0.2, Repo 6's threshold) pushes toward "more AI-like".
        if repetition_burstiness < 0.2:
            base_score = min(100, base_score + 10)
        elif repetition_burstiness > 0.5:
            base_score = max(0, base_score - 10)

        confidence = "medium" if 100 <= len(text) < 400 else "high"

        return EngineOutput(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            category=self.category,
            score=base_score,
            confidence=confidence,
            summary=(
                f"Sentence-level perplexity peak of {burstiness_ppl:.1f} and word-repetition "
                f"burstiness of {repetition_burstiness:.2f} suggest "
                f"{'AI-generated' if base_score >= 60 else 'human-written'} text."
            ),
            details={
                "full_text_perplexity": round(full_ppl, 2),
                "sentence_burstiness_perplexity": round(burstiness_ppl, 2),
                "word_repetition_burstiness": round(repetition_burstiness, 3),
                "sentences_analyzed": len(sentence_ppls),
            },
        )
