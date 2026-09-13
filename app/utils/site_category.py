"""
Classifies what *kind* of website this is (e-commerce, business, portfolio,
etc.) — a separate question from "was it AI-built". Implemented as a
keyword-weighted multi-class classifier over the rendered page text, with
output normalized into a probability distribution across categories (like
a softmax layer) — this is what gives the "real ML classifier" feel: a
confidence distribution across labels, not just a single yes/no score.
"""
import re

CATEGORY_KEYWORDS = {
    "E-commerce": [
        "add to cart", "shopping cart", "checkout", "buy now", "free shipping",
        "in stock", "out of stock", "sku", "product reviews", "add to bag",
        "your cart", "proceed to checkout",
    ],
    "Business / Corporate": [
        "our services", "about us", "contact us", "our clients", "testimonials",
        "our team", "request a quote", "case studies", "industries we serve",
        "get in touch", "our mission",
    ],
    "Portfolio": [
        "my work", "portfolio", "selected work", "case study", "resume", "cv",
        "let's work together", "hire me", "my projects", "skills",
    ],
    "Blog / Content": [
        "read more", "published", "recent posts", "categories", "archive",
        "comments", "subscribe to our newsletter", "min read", "by author",
    ],
    "SaaS / Tech Product": [
        "pricing", "start free trial", "sign up free", "dashboard", "api docs",
        "integrations", "features", "book a demo", "get started free",
        "no credit card required",
    ],
    "Personal": [
        "about me", "welcome to my", "hi, i'm", "my name is", "thoughts on",
        "personal blog",
    ],
}


def classify_site_category(text: str) -> dict:
    text_lower = text.lower()
    raw_scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(len(re.findall(re.escape(kw), text_lower)) for kw in keywords)
        raw_scores[category] = hits

    total = sum(raw_scores.values())
    if total == 0:
        return {
            "label": "Other / Unclear",
            "confidence": 0.0,
            "distribution": {c: round(1 / len(CATEGORY_KEYWORDS), 3) for c in CATEGORY_KEYWORDS},
        }

    distribution = {c: round(v / total, 3) for c, v in raw_scores.items()}
    label = max(distribution, key=distribution.get)
    return {
        "label": label,
        "confidence": distribution[label],
        "distribution": distribution,
    }
