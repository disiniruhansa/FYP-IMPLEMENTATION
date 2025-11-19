# models/response_generator.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class EmpatheticResponder:
    """
    Wrapper around a Flan-T5 model to generate short, empathetic responses
    for menstrual health conversations.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", device: str | None = None):
        print("[EmpatheticResponder] Loading model... This may take a moment the first time.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        print(f"[EmpatheticResponder] Model loaded on {self.device}.")

    def _build_prompt(self, user_message: str, emotions: list[str] | None = None) -> str:
        """
        Build an instruction-style prompt to guide the model
        to answer empathetically and safely.
        """
        emo_text = ", ".join(emotions) if emotions else "unknown"

        prompt = f"""
You are EmpowerHer, an emotionally sensitive menstrual health support chatbot.
Your job is to reply with a short, empathetic, supportive message.
Do NOT give medical diagnosis. Encourage the user to seek a doctor if needed.

User message:
"{user_message}"

Detected emotions: {emo_text}

Now write a warm, kind, and non-judgmental response (3–5 sentences).
        """.strip()

        return prompt

    def generate_response(
        self,
        user_message: str,
        emotions: list[str] | None = None,
        max_new_tokens: int = 120,
    ) -> str:
        """
        Generate an empathetic response for the given message and emotion list.
        """
        if not user_message or not user_message.strip():
            return "I'm here to listen whenever you'd like to share how you're feeling."

        prompt = self._build_prompt(user_message, emotions)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()


# Simple test
if __name__ == "__main__":
    responder = EmpatheticResponder()

    test_message = "I'm really scared because my period is very late and I don't know if this is normal."
    test_emotions = ["anxiety", "fear"]

    reply = responder.generate_response(test_message, test_emotions)
    print("\nUSER :", test_message)
    print("BOT  :", reply)
