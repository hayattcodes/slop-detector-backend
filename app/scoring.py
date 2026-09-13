"""
Combines every engine's individual score into one overall "granularity
score". A trained classifier (transformer_classifier) is treated as the
primary/trusted signal since it's validated across modern LLM families.
Heuristic engines (GPT-2 perplexity, etc.) act as supporting evidence —
they inform the reasons/flags but don't get equal vote in the final number,
since GPT-2-era perplexity is known to under-score modern AI text.

Tie-breaker logic (adapted from SlopTotal's "unanimous-high skepticism"
approach): when the trusted classifier is very confidently "AI" but the
linguistic/statistical engines strongly disagree, we pull the score down
slightly rather than trusting the classifier blindly — this catches the
"formal, well-structured human text mistaken for AI" false-positive case.
"""
from app.engines.base import EngineOutput

BANDS = [
    (30, "Clean"),
    (50, "Low Risk"),
    (70, "Suspicious"),
    (90, "Likely AI"),
    (101, "High Confidence AI"),
]

CONFIDENCE_MULTIPLIER = {"low": 0.5, "medium": 0.8, "high": 1.0}

TRUSTED_ENGINE_IDS = {"transformer_classifier"}
DISAGREEMENT_ENGINE_IDS = {"statistical_linguistic"}


def combine_scores(results: list[EngineOutput], weights: dict[str, float]) -> tuple[float, str]:
    if not results:
        return 0.0, "Clean"

    usable = [r for r in results if r.confidence != "low"]
    if not usable:
        usable = results  # fallback: nothing usable, avoid crashing

    weighted_sum = 0.0
    weight_total = 0.0
    for r in usable:
        w = weights.get(r.engine_id, 1.0) * CONFIDENCE_MULTIPLIER.get(r.confidence, 0.7)
        weighted_sum += r.score * w
        weight_total += w

    heuristic_avg = weighted_sum / weight_total if weight_total else 0.0

    trusted = [r for r in usable if r.engine_id in TRUSTED_ENGINE_IDS]
    if trusted:
        trusted_avg = sum(r.score for r in trusted) / len(trusted)
        overall = 0.8 * trusted_avg + 0.2 * heuristic_avg
    else:
        overall = heuristic_avg
        trusted_avg = None

    # --- Unanimous-high skepticism tie-breaker ---
    if trusted_avg is not None and trusted_avg >= 85:
        disagreement = [r for r in usable if r.engine_id in DISAGREEMENT_ENGINE_IDS]
        for d in disagreement:
            if d.score < 20:
                # Strong disagreement from a linguistic engine — pull the
                # score down by 15 points as a skepticism penalty, but
                # never below the "Suspicious" band floor (70).
                overall = max(70.0, overall - 15)

    band = "Clean"
    for threshold, label in BANDS:
        if overall < threshold:
            band = label
            break

    return round(overall, 1), band