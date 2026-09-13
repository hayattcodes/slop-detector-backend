"""
Given whatever the person typed/pasted into the single input box, decide
whether it's a URL or plain text. File uploads go through a separate
multipart endpoint (see routers/detect.py) since they can't share a text
input box.
"""
import re

_URL_PATTERN = re.compile(
    r"^(https?://)?"                          # optional scheme
    r"([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}"          # domain
    r"(:\d+)?(/[^\s]*)?$"                      # optional port + path
)


def classify_input(raw_value: str) -> str:
    value = raw_value.strip()
    # A GitHub repo link is handled by a dedicated engine (commit-pattern
    # analysis) rather than the generic website pipeline.
    if re.match(r"^(https?://)?(www\.)?github\.com/[^/\s]+/[^/\s?#]+/?$", value):
        return "github_repo"
    # Treat as a URL only if it's a single "word" (no spaces) that matches
    # a domain-like pattern — otherwise even something like "visit
    # example.com today" should be analyzed as text, not fetched as a URL.
    if " " not in value and _URL_PATTERN.match(value):
        return "url"
    return "text"


def normalize_url(raw_value: str) -> str:
    value = raw_value.strip()
    if not value.startswith(("http://", "https://")):
        value = "https://" + value
    return value
