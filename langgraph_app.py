from langgraph.graph import StateGraph, END
from inference_node import inference_node
from confidence_check_node import confidence_check_node
from fallback_node import fallback_node
from clarification_node import clarification_node
import subprocess

State = dict
builder = StateGraph(State)
builder.add_node("inference", inference_node)
builder.add_node("check_confidence", confidence_check_node)
builder.add_node("fallback_handler", fallback_node)
builder.add_node("clarify", clarification_node)

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
builder.add_conditional_edges(
    "fallback_handler",
    lambda state: "clarified_text" if state.get("backup_confidence", 1.0) < 0.7 else "done",
    path_map={
        "clarified_text": "clarify",
        "done": END
    }
)
builder.add_edge("clarify", "fallback_handler")
graph = builder.compile()

if __name__ == "__main__":
    print("\nStarting LangGraph inference pipeline with fallback and clarification support\n")
    input_text = input("Enter a text to classify: ")
    final_state = graph.invoke({"text": input_text})
    
    print("\nWORKFLOW SUMMARY:")
    print("\nUser Input:", final_state.get("text"))
    print("\nInference Prediction:", final_state.get("prediction"))
    print("\nConfidence Check Status:", final_state.get("status"))

    if final_state.get("fallback_triggered")==True:
        print("Fallback was used")
        print("Fallback model confidence score: ",final_state.get("backup_confidence"))

        if final_state.get("clarification_used")==True:
            #
            print("Clarification used before fallback")
            print("Clarified Text:", final_state.get("clarified_text"))
        print("\nFinal Label:", final_state.get("final_label"))
        print("Backup Confidence:", f"{final_state.get('backup_confidence', 0):.2f}")
    else:
        print("\nFallback not triggered")
        print("\nFinal Prediction:", final_state.get("prediction"))
        print("Confidence:", f"{final_state.get('confidence', 0):.2f}")

    
    subprocess.run(["python", "log_vizualizer.py"])
