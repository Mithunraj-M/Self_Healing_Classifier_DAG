# backup_fallback_node.py

import logging
from datetime import datetime
from transformers import pipeline

logger = logging.getLogger("fallback")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/fallback.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def fallback_node(state: dict) -> dict:
    print("\nConfidence too low Fallback node triggered: ")
    text = state.get("text", "")
    orig_prediction = state.get("prediction", "unknown")
    orig_confidence = state.get("confidence", 0.0)

    logger.warning(f"[BACKUP FALLBACK] Running zero-shot on: '{text}'")

    result = zero_shot(text, candidate_labels=LABELS)
    z_label = result["labels"][0]
    z_confidence = result["scores"][0]

    logger.info(f"[BACKUP RESULT] Text: '{text}' -> Zero-Shot Label: {z_label} (Confidence: {z_confidence:.4f})")
    print("Fallback Model Prediction:",z_label)
    print("Fallback Model Confidence Score: ",z_confidence)
    return {
        **state,
        "fallback_triggered": True,
        "backup_model": "facebook/bart-large-mnli",
        "corrected_label": z_label,
        "final_label": z_label,
        "backup_confidence": z_confidence,
        "fallback_count": state.get("fallback_count", 0) + 1,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    test_input = {
        "text": "I feel like nobody understands me anymore.",
        "prediction": "fear",
        "confidence": 0.45,
        "status": "fallback"
    }
    updated = fallback_node(test_input)
    print(" Final State:", updated)
