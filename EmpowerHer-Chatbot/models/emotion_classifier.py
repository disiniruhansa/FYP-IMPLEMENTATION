# models/emotion_classifier.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


class EmotionClassifier:
    """
    Wrapper around a pre-trained GoEmotions RoBERTa model.
    Uses: SamLowe/roberta-base-go_emotions
    """

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        print("[EmotionClassifier] Loading model... This may take a moment the first time.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # text-classification pipeline does multi-label prediction for this model
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )
        print("[EmotionClassifier] Model loaded successfully.")

    def predict_emotions(self, text: str, top_k: int = 3):
        """
        Predict top_k emotions for a given input text.
        Returns a list of {label, score} dicts.
        """
        if not text or not text.strip():
            return []

        # Get scores for all labels
        outputs = self.pipeline(text)[0]  # pipeline returns a list per input
        # Sort by score descending
        sorted_outputs = sorted(outputs, key=lambda x: x["score"], reverse=True)
        # Take top_k
        return sorted_outputs[:top_k]


# Simple test code so you can run this file directly
if __name__ == "__main__":
    clf = EmotionClassifier()

    test_sentences = [
        "I feel really scared because my period is very late.",
        "I'm so happy that my cycle is finally regular.",
        "I'm embarrassed to talk about my period with anyone."
    ]

    for s in test_sentences:
        print(f"\nText: {s}")
        preds = clf.predict_emotions(s, top_k=5)
        for p in preds:
            print(f"  {p['label']}: {p['score']:.3f}")
