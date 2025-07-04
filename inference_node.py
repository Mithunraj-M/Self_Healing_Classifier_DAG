from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import numpy as np
import logging


logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/inference.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

MODEL_DIR = "emotion_lora_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

class_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label = class_labels[pred_idx]

    logger.info(f"Text: '{text}' | Prediction: {label} | Confidence: {confidence:.4f}")
    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "probs": probs.tolist()
    }


def inference_node(state: dict) -> dict:
    print("Inference Node Activated:")
    input_text = state.get("text")
    if not input_text:
        raise ValueError("Missing 'text' in state for inference.")

    result = predict(input_text)

    print("User Input: ",input_text)
    print("Inference Prediction: ",{result["label"]})
    print("Confidence Score: ",{result["confidence"]})
    print("Probabality: ",result["probs"])
    return {
        **state,
        "prediction": result["label"],
        "confidence": result["confidence"],
        "probs": result["probs"]
    }


if __name__ == "__main__":
    test_state = {"text": "I'm feeling very nervous about this new job."}
    updated_state = inference_node(test_state)
    print(updated_state)
