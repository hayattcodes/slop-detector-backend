"""
Finds spans of text worth highlighting for the user — known AI-cliché
phrases and (for websites) AI-builder signature phrases — with their
character offsets in the original text, so the frontend can render them
inline (like a grammar-checker highlighting suspicious passages). This is
the explainability feature Repo 1 (IviweBooi) called "suspicious text
highlighting".
"""
import re

from app.engines.statistical_linguistic import AI_CLICHE_PHRASES
from app.engines.website_fingerprint import SIGNATURE_PHRASES, PLACEHOLDER_PHRASES


def find_highlights(text: str) -> list[dict]:
    highlights = []
    text_lower = text.lower()

    for phrase in AI_CLICHE_PHRASES:
        for match in re.finditer(re.escape(phrase), text_lower):
            highlights.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "phrase": text[match.start():match.end()],
                    "reason": "Common AI-generated phrasing",
                }
            )

    for phrase in SIGNATURE_PHRASES + PLACEHOLDER_PHRASES:
        for match in re.finditer(re.escape(phrase), text_lower):
            highlights.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "phrase": text[match.start():match.end()],
                    "reason": "AI-builder signature / placeholder text",
                }
            )

    highlights.sort(key=lambda h: h["start"])
    # Drop overlaps (keep the first/longest match found at each position)
    merged = []
    last_end = -1
    for h in highlights:
        if h["start"] >= last_end:
            merged.append(h)
            last_end = h["end"]
    return merged
