"""
Transformer Classifier engine — a real, pretrained AI-text-detection model.

Uses SuperAnnotate/ai-detector (RoBERTa-large fine-tuned), trained on 14
different LLMs across the GPT, LLaMA, Anthropic (Claude), and Mistral
families, with published benchmark accuracy: 98.5% on GPT-4, 99.2% on
ChatGPT, 98.0% on LLaMA-Chat (RAID benchmark). This is a meaningfully
better fit for today's AI-generated text than older GPT-2-era detectors
(like the original OpenAI roberta-base-openai-detector), whose own model
card explicitly warns it gets less accurate against newer/larger models —
which is exactly why it was under-scoring modern ChatGPT/Claude-style text.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from generated_text_detector.utils.preprocessing import preprocessing_text
from transformers import AutoTokenizer

from app.engines.base import Engine, EngineOutput

MODEL_ID = "SuperAnnotate/ai-detector"


@lru_cache(maxsize=1)
def _load_model():
    model = RobertaClassifier.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer


class TransformerClassifierEngine(Engine):
    engine_id = "transformer_classifier"
    engine_name = "Transformer Classifier (SuperAnnotate AI Detector)"
    category = "content"
    weight = 1.0  # a trained classifier is generally more reliable than heuristics

    def applies_to(self, input_type: str) -> bool:
        return input_type in ("text", "url", "file")

    def run(self, *, text: str = "", html: str = "", url: str = "", raw_html_before_js: str = "") -> EngineOutput:
        text = text.strip()
        if len(text) < 50:
            return EngineOutput(
                engine_id=self.engine_id,
                engine_name=self.engine_name,
                category=self.category,
                score=0.0,
                confidence="low",
                summary="Text is too short for the transformer classifier to give a reliable reading.",
                details={"reason": "insufficient_length"},
            )

        model, tokenizer = _load_model()
        cleaned = preprocessing_text(text[:4000])
        tokens = tokenizer(
            cleaned,
            add_special_tokens=True,
            max_length=512,
            padding="longest",
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            _, logits = model(**tokens)
        ai_probability = F.sigmoid(logits).squeeze(1).item() * 100

        confidence = "high" if len(text) > 200 else "medium"

        return EngineOutput(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            category=self.category,
            score=ai_probability,
            confidence=confidence,
            summary=f"SuperAnnotate AI Detector estimates {ai_probability:.0f}% probability of AI-generated text.",
            details={
                "model": MODEL_ID,
                "benchmark_note": "Trained across GPT/LLaMA/Anthropic/Mistral families (RAID benchmark avg. 85.2%)",
                "text_truncated_to_chars": len(cleaned) if len(cleaned) < len(text) else None,
            },
        )