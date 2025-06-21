# confidence_check_node.py



# Default threshold
CONFIDENCE_THRESHOLD = 0.7

# This function will act as a node in the LangGraph
# It receives the state from inference and decides whether fallback is needed
def confidence_check_node(state: dict) -> str:
    confidence = state.get("confidence", 0.0)
    label = state.get("label", "")
    text = state.get("text", "")

    status = "fallback" if confidence < CONFIDENCE_THRESHOLD else "accept"

    # Logging
    with open("logs/confidence_check.log", "a") as log:
        log.write(f"[CHECK] Text: '{text}' | Label: {label} | Confidence: {confidence:.4f} | Status: {status}\n")

    return {
        **state,
        "status": status  # ✅ Added to track decision in state
    }

# For unit testing or CLI logic
if __name__ == "__main__":
    test = {"text": "I'm scared.", "label": "fear", "confidence": 0.45}
    outcome = confidence_check_node(test)
    print("Decision:", outcome)
