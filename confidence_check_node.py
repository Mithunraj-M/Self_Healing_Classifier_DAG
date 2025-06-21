CONFIDENCE_THRESHOLD = 0.7

def confidence_check_node(state: dict) -> str:
    print("\nConfidence Check Node Activated:")
    confidence = state.get("confidence", 0.0)
    label = state.get("label", "")
    text = state.get("text", "")

    status = "fallback" if confidence < CONFIDENCE_THRESHOLD else "accept"

    with open("logs/confidence_check.log", "a") as log:
        log.write(f"[CHECK] Text: '{text}' | Label: {label} | Confidence: {confidence:.4f} | Status: {status}\n")
    print("Confidence Check Node Output Status: ",status)
    return {
        **state,
        "status": status  
    }


if __name__ == "__main__":
    test = {"text": "I'm scared.", "label": "fear", "confidence": 0.45}
    outcome = confidence_check_node(test)
    print("Decision:", outcome)
