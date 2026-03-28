"""
Quick prompt optimization evaluation for the EmpatheticResponder.

Usage:
  python -m eval.prompt_eval

Output:
  - eval/prompt_eval_results.csv
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

from models.response_generator import EmpatheticResponder


@dataclass
class PromptCase:
    user_message: str
    emotions: list[str]
    retrieved_context: str
    category: str


CASES = [
    PromptCase(
        user_message="I'm scared because my period is late.",
        emotions=["fear", "anxiety"],
        retrieved_context="A period can come late because of stress, sleep changes, illness, or normal hormone shifts. Seek help if there is severe pain, fever, dizziness, or very heavy bleeding.",
        category="late_period",
    ),
    PromptCase(
        user_message="How can I reduce cramps naturally?",
        emotions=["curiosity", "worry"],
        retrieved_context="Gentle heat, rest, warm drinks, light stretching, and slow walking may help cramps. Severe pain or pain that stops normal activity should be checked by a doctor.",
        category="cramps",
    ),
    PromptCase(
        user_message="I feel itchy and there is a bad smell.",
        emotions=["worry", "disgust"],
        retrieved_context="A strong smell with itching, burning, unusual discharge, or fever should be checked by a clinic. Gentle washing and regular pad changes can help freshness.",
        category="odor_discharge",
    ),
    PromptCase(
        user_message="for 10 days",
        emotions=["worry"],
        retrieved_context="A late period can happen because of stress, illness, travel, or hormone changes. If the delay continues with severe pain, heavy bleeding, dizziness, or fever, seek medical help.",
        category="follow_up_fragment",
    ),
    PromptCase(
        user_message="I love you",
        emotions=["love"],
        retrieved_context="",
        category="out_of_scope",
    ),
    PromptCase(
        user_message="I'm angry and sad before my period.",
        emotions=["anger", "sadness"],
        retrieved_context="Mood changes before a period are common. Rest, support, and talking to someone trusted may help. Seek help if emotions feel overwhelming.",
        category="mood_support",
    ),
]


def main() -> None:
    responder = EmpatheticResponder()
    output_path = Path("eval") / "prompt_eval_results.csv"

    rows: list[dict[str, str]] = []
    for case in CASES:
        reply = responder.generate(
            case.user_message,
            emotions=case.emotions,
            retrieved_context=case.retrieved_context or None,
        )
        rows.append(
            {
                "category": case.category,
                "user_message": case.user_message,
                "emotions": "|".join(case.emotions),
                "retrieved_context": case.retrieved_context,
                "reply": reply,
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "user_message", "emotions", "retrieved_context", "reply"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved prompt evaluation results to: {output_path}")


if __name__ == "__main__":
    main()
