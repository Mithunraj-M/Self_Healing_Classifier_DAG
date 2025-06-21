from langgraph.graph import StateGraph, END
from inference_node import inference_node
from confidence_check_node import confidence_check_node
from fallback_node import fallback_node
import subprocess
State = dict
builder = StateGraph(State)
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
graph = builder.compile()

if __name__ == "__main__":
    
    print("\n Starting LangGraph inference pipeline with fallback support\n")
    input_text = input("Enter a text to classify: ")
    final_state = graph.invoke({"text": input_text})
    print("\nWORKFLOW SUMMARY:")
    print("\nUser Input: ",{final_state['text']})
    print("Inference prediction",{final_state['prediction']})
    print
    print("\nConfidence check status: ",{final_state['status']})
    
    if final_state['status'] != 'accept':
        print("Fallback triggered: zero-shot-model")
        print(f"\nFinal prediction: {final_state['final_label']}")
        print(f"Confidence: {final_state['backup_confidence']:.2f}")
    else:
        print("Fallback not triggered")
        print(f"\nFinal Prediction: {final_state['prediction']}")
        print(f"Confidence: {final_state['confidence']:.2f}")
    
    
    if final_state.get("fallback_triggered"):
        print("(Fallback was used)")
    subprocess.run(["python", "log_vizualizer.py"])