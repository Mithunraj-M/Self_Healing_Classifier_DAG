CONFIDENCE_THRESHOLD = 0.7

CONFIDENCE_THRESHOLD = 0.7

def confidence_check_node(state: dict) -> dict:
    print("\nConfidence Check Node Activated:")
    confidence = state.get("confidence", 0.0)
    label = state.get("prediction", "")
    text = state.get("text", "")

    status = "fallback" if confidence < 0.7 else "accept"

    with open("logs/confidence_check.log", "a", encoding="utf-8") as log:
        log.write(f"[CHECK] Text: '{text}' | Label: {label} | Confidence: {confidence:.4f} | Status: {status}\n")

    print("Confidence Check Node Output Status:", status)
    return {
        **state,
        "status": status
    }




if __name__ == "__main__":
    test = {"text": "I'm scared.", "label": "fear", "confidence": 0.45}
    outcome = confidence_check_node(test)
    print("Decision:", outcome)
