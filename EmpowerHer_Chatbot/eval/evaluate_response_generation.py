import os
import pandas as pd
from services.chat_service import ChatService
from models.emotion_classifier import EmotionClassifier

# same bucket mapping you used in chat_service.py
def emotion_bucket(labels):
    labels = [l.lower() for l in (labels or [])]
    if any(l in labels for l in ["fear", "anxiety", "nervousness", "worry"]):
        return "anxious"
    if any(l in labels for l in ["sadness", "disappointment", "grief", "loneliness", "remorse"]):
        return "sad"
    if any(l in labels for l in ["anger", "annoyance", "irritation", "disapproval"]):
        return "angry"
    return "mixed"

def top_labels(preds):
    if not isinstance(preds, list):
        return []
    return [p.get("label", "") for p in preds if isinstance(p, dict)]

def main():
    preferred = "eval/dg_eval.csv"
    fallback = "eval/emotion_eval.csv"
    csv_path = preferred if os.path.exists(preferred) else fallback

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "user_message" not in df.columns:
        if "text" in df.columns:
            df = df.rename(columns={"text": "user_message"})
        elif len(df.columns) > 0:
            first = df.columns[0]
            df = df.rename(columns={first: "user_message"})
            print(f"[warn] 'user_message' column missing; using first column '{first}'.")
        else:
            raise ValueError("Input CSV has no columns; expected a 'user_message' column.")

    chat = ChatService()
    clf = EmotionClassifier()

    rows = []
    for _, r in df.iterrows():
        user = str(r["user_message"])

        # chatbot reply
        result = chat.generate_reply(user)
        reply = result.reply

        # emotion bucket from user
        user_preds = clf.predict_emotions(user, top_k=3)
        user_bucket = emotion_bucket(top_labels(user_preds))

        # emotion bucket from bot reply
        bot_preds = clf.predict_emotions(reply, top_k=3)
        bot_bucket = emotion_bucket(top_labels(bot_preds))

        match = int(user_bucket == bot_bucket)

        rows.append({
            "user_message": user,
            "reply": reply,
            "user_emotions": "|".join(top_labels(user_preds)),
            "bot_emotions": "|".join(top_labels(bot_preds)),
            "user_bucket": user_bucket,
            "bot_bucket": bot_bucket,
            "emotion_match": match,
            "intent": result.intent,
            "topic": result.topic
        })

    out = pd.DataFrame(rows)
    out.to_csv("eval/dg_results.csv", index=False)

    # Report
    print("\nSaved: eval/dg_results.csv")
    print("Emotion Match Rate:", out["emotion_match"].mean())
    print("\nBy intent:")
    print(out.groupby("intent")["emotion_match"].mean())

if __name__ == "__main__":
    main()
