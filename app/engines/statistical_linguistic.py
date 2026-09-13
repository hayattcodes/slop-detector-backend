"""
Statistical & Linguistic Heuristics engine — adapted from the non-neural
half of Repo 4 (SlopTotal)'s 23-engine ensemble. These are cheap,
explainable signals that don't need any model download, and catch
different things than perplexity/burstiness does:
- lexical diversity (AI text tends to reuse a narrower vocabulary)
- sentence-length uniformity (AI text is often suspiciously consistent;
  human writing varies sentence length more)
- "AI cliché" phrase detection (a known list of phrases that show up
  disproportionately often in LLM output: "delve into", "in today's
  fast-paced world", "it's important to note", etc.)
- em-dash / transition-word overuse (a widely observed LLM stylistic tic)
"""
import re
import statistics

from app.engines.base import Engine, EngineOutput

AI_CLICHE_PHRASES = [
    "delve into", "navigate the complexities", "in today's fast-paced world",
    "it's important to note", "it is important to note", "in conclusion",
    "furthermore", "moreover", "as an ai language model", "in the realm of",
    "plays a pivotal role", "plays a crucial role", "unlock the potential",
    "unleash the power", "in this ever-evolving landscape", "testament to",
    "boasts a", "stands as a", "rich tapestry", "let's dive in",
    "at the end of the day", "when it comes to", "it is worth noting",
]

TRANSITION_WORDS = [
    "furthermore", "moreover", "additionally", "consequently", "however",
    "nevertheless", "therefore", "thus", "hence", "accordingly",
]


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _lexical_diversity(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(w.lower() for w in words)) / len(words)


class StatisticalLinguisticEngine(Engine):
    engine_id = "statistical_linguistic"
    engine_name = "Statistical & Linguistic Patterns"
    category = "content"
    weight = 0.7  # cheap heuristics, moderate trust until calibrated

    def applies_to(self, input_type: str) -> bool:
        return input_type in ("text", "url", "file")

    def run(self, *, text: str = "", html: str = "", url: str = "", raw_html_before_js: str = "") -> EngineOutput:
        text = text.strip()
        if len(text) < 80:
            return EngineOutput(
                engine_id=self.engine_id,
                engine_name=self.engine_name,
                category=self.category,
                score=0.0,
                confidence="low",
                summary="Text is too short for reliable linguistic-pattern analysis.",
                details={"reason": "insufficient_length"},
            )

        sentences = _sentence_split(text)
        words = re.findall(r"[A-Za-z']+", text)
        points = 0.0
        max_points = 0.0
        signals = {}

        # 1. Lexical diversity — low diversity (narrow vocabulary) is more AI-like
        max_points += 25
        diversity = _lexical_diversity(words)
        if diversity < 0.35:
            diversity_points = 25
        elif diversity < 0.5:
            diversity_points = 12
        else:
            diversity_points = 0
        points += diversity_points
        signals["lexical_diversity"] = round(diversity, 3)

        # 2. Sentence-length uniformity — low variance (very consistent
        # sentence lengths) is more AI-like; human writing is bursty here too
        max_points += 25
        lengths = [len(s.split()) for s in sentences if s.split()]
        if len(lengths) >= 4:
            mean_len = statistics.mean(lengths)
            stdev_len = statistics.pstdev(lengths)
            coefficient_of_variation = (stdev_len / mean_len) if mean_len else 0
            if coefficient_of_variation < 0.3:
                uniformity_points = 25
            elif coefficient_of_variation < 0.5:
                uniformity_points = 12
            else:
                uniformity_points = 0
            signals["sentence_length_variation"] = round(coefficient_of_variation, 3)
        else:
            uniformity_points = 0
            signals["sentence_length_variation"] = None
        points += uniformity_points

        # 3. AI-cliché phrase density
        max_points += 30
        text_lower = text.lower()
        found_cliches = [p for p in AI_CLICHE_PHRASES if p in text_lower]
        cliche_points = min(30, len(found_cliches) * 10)
        points += cliche_points
        signals["ai_cliche_phrases_found"] = found_cliches

        # 4. Em-dash overuse (per 500 words)
        max_points += 10
        em_dash_count = text.count("—") + text.count("--")
        em_dash_rate = em_dash_count / (len(words) / 500) if words else 0
        em_dash_points = 10 if em_dash_rate > 3 else (5 if em_dash_rate > 1.5 else 0)
        points += em_dash_points
        signals["em_dash_rate_per_500_words"] = round(em_dash_rate, 2)

        # 5. Transition-word overuse
        max_points += 10
        transition_count = sum(text_lower.count(t) for t in TRANSITION_WORDS)
        transition_rate = transition_count / (len(words) / 500) if words else 0
        transition_points = 10 if transition_rate > 4 else (5 if transition_rate > 2 else 0)
        points += transition_points
        signals["transition_word_rate_per_500_words"] = round(transition_rate, 2)

        overall_score = (points / max_points) * 100 if max_points else 0.0
        confidence = "high" if len(words) > 150 else "medium"

        summary_bits = []
        if found_cliches:
            summary_bits.append(f"{len(found_cliches)} common AI-phrase(s) found")
        summary_bits.append(f"lexical diversity {diversity:.0%}")
        if signals["sentence_length_variation"] is not None:
            summary_bits.append(f"sentence-length variation {signals['sentence_length_variation']:.2f}")
        summary = "; ".join(summary_bits)

        signals["score_breakdown"] = {
            "lexical_diversity": diversity_points,
            "sentence_uniformity": uniformity_points,
            "ai_cliche_phrases": cliche_points,
            "em_dash_overuse": em_dash_points,
            "transition_word_overuse": transition_points,
        }

        return EngineOutput(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            category=self.category,
            score=overall_score,
            confidence=confidence,
            summary=summary,
            details=signals,
        )
