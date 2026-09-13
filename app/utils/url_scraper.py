"""
Fetches a URL and splits the response into what engines need:
- `html`: the fully rendered HTML (after JavaScript has run) — this matters
  a lot for AI-website-builder detection, because tools like Lovable/Bolt/v0
  ship React/Next single-page apps: the raw server response is often just
  an empty <div id="root"></div> shell, with all the actual markup, classes,
  and content injected by JavaScript at runtime. A plain `requests.get()`
  never sees any of that — which was the root cause of consistently low
  Website Fingerprint scores. We use a real (headless) browser instead, so
  we see exactly what a person's browser would see.
- `text`: the readable, visible text content (for the content engines)
- `raw_html_before_js`: the original un-rendered server response, kept
  around for informational purposes (some signature comments only appear
  in the raw source, e.g. server-side generator tags)
"""
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests

REQUEST_TIMEOUT_MS = 20_000
USER_AGENT = "Mozilla/5.0 (compatible; AIDetectorBot/0.1; +research tool)"


class ScrapeError(Exception):
    pass


def _fetch_raw_html(url: str) -> str:
    """Best-effort fetch of the un-rendered server response, used only as
    a fallback source for signature phrases. Never raises — if it fails,
    we still have the rendered HTML from Playwright."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
        return resp.text
    except requests.RequestException:
        return ""


def scrape_url(url: str) -> dict:
    raw_html_before_js = _fetch_raw_html(url)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(user_agent=USER_AGENT)
            # "load" is more reliable than "networkidle" — many sites keep
            # background network activity going forever (analytics, chat
            # widgets, polling) which means "networkidle" never fires and
            # the whole scan times out even though the page rendered fine.
            page.goto(url, timeout=REQUEST_TIMEOUT_MS, wait_until="load")
            # Small extra pause so React/Next SPAs finish their initial
            # client-side render after the load event fires.
            page.wait_for_timeout(1500)
            rendered_html = page.content()
            browser.close()
    except Exception as e:
        raise ScrapeError(
            f"Could not render the page (is the URL correct and publicly reachable?): {e}"
        ) from e

    soup = BeautifulSoup(rendered_html, "lxml")
    title = soup.title.string if soup.title else ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)

    return {
        "html": rendered_html,
        "raw_html_before_js": raw_html_before_js,
        "text": text,
        "title": title,
    }