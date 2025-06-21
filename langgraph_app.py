

from langgraph.graph import StateGraph, END
from inference_node import inference_node
from confidence_check_node import confidence_check_node
from fallback_node import fallback_node


State = dict

# Create the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("inference", inference_node)
builder.add_node("check_confidence", confidence_check_node)
builder.add_node("fallback_handler", fallback_node)


builder.set_entry_point("inference")
builder.add_edge("inference", "check_confidence")
builder.add_conditional_edges(
    "check_confidence",
    lambda state: state.get("status"),
    path_map={
        "accept": END,
        "fallback": "fallback_handler"
    }
)
builder.add_edge("fallback_handler", END)

# Compile it
graph = builder.compile()

# --- Example run ---
if __name__ == "__main__":
    print("\n🧪 Starting LangGraph inference pipeline with fallback support\n")
    input_text = input("Enter a text to classify: ")
    final_state = graph.invoke({"text": input_text})

    print("\n🎯 Final Prediction:")
    print(f"Text: {final_state['text']}")
    print(f"Final Label: {final_state['final_label']}")
    print(f"Confidence: {final_state['confidence']:.2f}")
    if final_state.get("fallback_triggered"):
        print("(Fallback was used)")
