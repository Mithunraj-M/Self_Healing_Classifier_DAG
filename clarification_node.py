import logging
from datetime import datetime
from typing import Dict
import openai
import os
from dotenv import load_dotenv

logger = logging.getLogger("clarification")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/clarification.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

CLARIFICATION_PROMPT = """
Your task is to clarify the user's emotion based on a vague input.

User said: "{text}"

Ask a short follow-up question that helps identify their emotion.
Examples:
- What about that experience made you feel this way?
- Can you describe how it made you feel?
- Was it a positive or negative emotion?
"""

AUGMENTATION_PROMPT = """
You are an assistant helping clarify user emotions.

Original input:
"{text}"

User's clarification:
"{clarification}"

Now rewrite the original input into a clearer sentence that includes the user's emotional expression.
Return a single clarified sentence.
"""

def clarification_node(state: Dict) -> Dict:
    original_text = state.get("text")
    prompt = CLARIFICATION_PROMPT.format(text=original_text)
    logger.info(f"Triggering clarification for input: {original_text}")
    load_dotenv()

    try:
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = os.getenv("OPENROUTER_API_KEY")

        
        clarification_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        clarification_question = clarification_response["choices"][0]["message"]["content"].strip()

        print("\n Clarifier Bot:", clarification_question)
        user_input = input("You: ").strip()

        
        rewrite_prompt = AUGMENTATION_PROMPT.format(text=original_text, clarification=user_input)
        rewriting_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": rewrite_prompt}
            ]
        )
        clarified_text = rewriting_response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logger.error(f"LLM clarification failed: {e}")
        clarified_text = original_text + " (clarification failed)"

    logger.info(f"Clarified input: {clarified_text}")
    return {
        **state,
        "clarified_text": clarified_text,
        "text": clarified_text,
        "timestamp": datetime.utcnow().isoformat(),
        "clarification_used": True
    }

if __name__ == "__main__":
    test = {"text": "I got first place in hackathon"}
    updated = clarification_node(test)
    print("\nClarified State:", updated)
