# models/empathetic_templates.py

from __future__ import annotations
from typing import List, Dict
import random

# High-level emotion buckets for menstrual support
EMOTION_BUCKETS: Dict[str, List[str]] = {
    "fear": [
        "It's completely understandable to feel scared when your body or period changes.",
        "Feeling worried about your period is very normal, and you’re not alone in this.",
        "Many girls feel anxious when something about their cycle feels different, and that’s okay."
    ],
    "sadness": [
        "I'm really sorry that you're feeling low right now; your feelings truly matter.",
        "It's okay to feel sad or down when you're going through changes with your body.",
        "Feeling tearful or emotional around your period is very common, and it doesn’t mean anything is wrong with you."
    ],
    "anger": [
        "Feeling angry or easily annoyed before or during your period is very normal.",
        "It makes sense that you're frustrated – hormone changes can make emotions feel much stronger.",
        "Being snappy or irritated around this time doesn't make you a bad person; it's part of how some bodies react."
    ],
    "stress": [
        "It sounds like you’re carrying a lot on your mind right now, and that can feel really heavy.",
        "School, family and body changes together can feel like too much sometimes, and that’s completely valid.",
        "You’re doing your best in a stressful situation, and that’s something to be proud of."
    ],
    "physical_discomfort": [
        "Cramps and other period pains can be really uncomfortable, and it’s okay to feel upset about it.",
        "Many girls experience strong cramps, bloating, or back pain during their period – you’re not alone in that.",
        "It’s completely valid to feel tired or drained when your body is going through menstrual pain."
    ],
    "neutral_support": [
        "Thank you for sharing how you feel – it’s brave to talk about this.",
        "You don’t have to go through this alone; it’s okay to reach out for support.",
        "Your feelings are important, and it’s good that you’re listening to your body."
    ],
}

# Map raw GoEmotions labels -> our buckets
RAW_TO_BUCKET: Dict[str, str] = {
    # fear / anxiety type
    "fear": "fear",
    "nervousness": "fear",
    "anxiety": "fear",
    "worry": "fear",

    # sadness
    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "loneliness": "sadness",

    # anger / irritation
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "irritation": "anger",

    # stress / overwhelm
    "confusion": "stress",
    "embarrassment": "stress",
    "shame": "stress",
    "guilt": "stress",

    # if nothing fits we’ll fall back to neutral_support
}


def map_raw_emotions_to_bucket(raw_labels: List[str]) -> str:
    """
    Take top-k emotion labels from EmotionClassifier and
    collapse them into one high-level bucket.
    """
    for label in raw_labels:
        bucket = RAW_TO_BUCKET.get(label)
        if bucket:
            return bucket

    # If nothing matched but we saw *any* emotion, use neutral_support
    if raw_labels:
        return "neutral_support"

    # No emotions? still neutral_support
    return "neutral_support"


def choose_template(bucket: str) -> str:
    """
    Randomly choose a human-written sentence from the given bucket.
    """
    options = EMOTION_BUCKETS.get(bucket) or EMOTION_BUCKETS["neutral_support"]
    return random.choice(options)
