# eval/evaluate_emotion_classifier.py
# Run from project root:
#   python -m eval.evaluate_emotion_classifier

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.emotion_classifier import EmotionClassifier


@dataclass
class EvalResult:
    top1_acc: float
    top3_acc: float
    top5_acc: float


def topk_hit(pred_labels: List[str], true_label: str, k: int) -> bool:
    return true_label in pred_labels[:k]


def normalize_label(x: str) -> str:
    return (x or "").strip().lower()


def main():
    # --- 1) Load your evaluation CSV ---
    # Create: eval/emotion_eval.csv with columns: text,true_label
    csv_path = os.path.join("eval", "emotion_eval.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Missing file: {csv_path}")
        print("Create eval/emotion_eval.csv with columns: text,true_label")
        return

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "true_label" not in df.columns:
        print("[ERROR] CSV must have columns: text,true_label")
        return

    texts = df["text"].astype(str).tolist()
    true_labels = [normalize_label(x) for x in df["true_label"].astype(str).tolist()]

    # --- 2) Load classifier ---
    clf = EmotionClassifier()

    # --- 3) Predict + compute Top-k accuracy ---
    pred_top1 = []
    top3_hits = 0
    top5_hits = 0

    for text, true_lab in zip(texts, true_labels):
        preds = clf.predict_emotions(text, top_k=5)

        # preds looks like: [{'label': 'fear', 'score': 0.88}, ...]
        pred_labels = [normalize_label(p.get("label", "")) for p in preds]

        if pred_labels:
            pred_top1.append(pred_labels[0])
        else:
            pred_top1.append("")

        if topk_hit(pred_labels, true_lab, 3):
            top3_hits += 1
        if topk_hit(pred_labels, true_lab, 5):
            top5_hits += 1

    top1 = accuracy_score(true_labels, pred_top1)
    top3 = top3_hits / len(true_labels)
    top5 = top5_hits / len(true_labels)

    print("\n========== Emotion Classifier Evaluation ==========")
    print(f"Samples: {len(true_labels)}")
    print(f"Top-1 Accuracy: {top1:.3f}")
    print(f"Top-3 Accuracy: {top3:.3f}")
    print(f"Top-5 Accuracy: {top5:.3f}")

    print("\n--- Classification Report (Top-1) ---")
    print(classification_report(true_labels, pred_top1, zero_division=0))

    print("\n--- Confusion Matrix (Top-1) ---")
    labels_sorted = sorted(list(set(true_labels)))
    cm = confusion_matrix(true_labels, pred_top1, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
