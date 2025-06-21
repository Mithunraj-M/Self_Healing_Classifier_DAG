import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger("fallback")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/fallback.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def fallback_node(state: dict) -> dict:
    text = state.get("text", "")
    prediction = state.get("prediction", "unknown")
    confidence = state.get("confidence", 0.0)
    status = state.get("status", "accept")  # default to accept if missing

    if status == "accept":
        print(f"\n✅ Accepted: '{text}' → {prediction} (Confidence: {confidence:.2f})")
        return {
            **state,
            "final_label": prediction,
            "fallback_triggered": False,
            "timestamp": datetime.utcnow().isoformat()
        }

    # Fallback triggered
    logger.warning(f"[FALLBACK] Low confidence ({confidence:.4f}) for: '{text}' → Predicted: '{prediction}'")

    print(f"\n⚠️ Fallback Triggered ⚠️")
    print(f"Input: {text}")
    print(f"Model Prediction: {prediction} (Confidence: {confidence:.2f})")
    print("Please enter the correct label from:", LABELS)

    corrected_label = None
    while corrected_label not in LABELS:
        corrected_label = input("Correct label: ").strip().lower()

    logger.info(f"User correction: '{text}' → {corrected_label}")

    return {
        **state,
        "fallback_triggered": True,
        "corrected_label": corrected_label,
        "final_label": corrected_label,
        "fallback_count": state.get("fallback_count", 0) + 1,
        "timestamp": datetime.utcnow().isoformat()
    }

# Test
if __name__ == "__main__":
    test_state = {
        "text": "I feel so alone today.",
        "prediction": "sadness",
        "confidence": 0.45,
        "status": "fallback"
    }
    updated = fallback_node(test_state)
    print("\n✅ Final State:", updated)
