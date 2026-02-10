# services/chat_service.py

from dataclasses import dataclass
from typing import List, Any
import random
import re

from models.emotion_classifier import EmotionClassifier
from services.kb_retriever import KnowledgeBaseRetriever, clean_kb_text


@dataclass
class ChatResult:
    reply: str
    emotions: List[str]
    raw_emotions: Any
    topic: str
    intent: str
    kb_sources: List[str]


# -------------------------------
# Emotion bucket (high-level)
# -------------------------------
def emotion_bucket(labels: List[str]) -> str:
    labels = [l.lower() for l in (labels or [])]

    if any(l in labels for l in ["fear", "anxiety", "nervousness", "worry"]):
        return "anxious"
    if any(l in labels for l in ["sadness", "disappointment", "grief", "loneliness", "remorse"]):
        return "sad"
    if any(l in labels for l in ["anger", "annoyance", "irritation", "disapproval"]):
        return "angry"
    return "mixed"


EMOTION_TEMPLATES = {
    "anxious": [
        "I can hear how worried you are, and it makes complete sense to feel that way.",
        "That sounds really scary right now, and your feelings are completely valid.",
        "It is okay to feel anxious — you are not alone in this.",
    ],
    "sad": [
        "I am really sorry you are feeling this way — that sounds heavy on you.",
        "It is okay to feel low sometimes, especially when your body feels confusing.",
        "Thank you for sharing this. Your feelings matter.",
    ],
    "angry": [
        "It is understandable to feel angry or frustrated — your feelings matter.",
        "That irritation makes sense, especially if you are in pain or stressed.",
        "You are not too much — this can be a tough time for many girls.",
    ],
    "mixed": [
        "Thank you for opening up — it is really brave to share how you feel.",
        "It is okay to have mixed feelings. You are doing your best.",
        "I am here with you. Let us take this one step at a time.",
    ],
}


def choose_emotion_line(bucket: str) -> str:
    return random.choice(EMOTION_TEMPLATES.get(bucket, EMOTION_TEMPLATES["mixed"]))


# -------------------------------
# Intent detector
# -------------------------------
def detect_intent(msg: str) -> str:
    m = msg.lower().strip()

    if any(p in m for p in [
        "calm down", "help me calm", "panic", "panicking", "anxiety attack",
        "can't breathe", "cant breathe", "overthinking"
    ]):
        return "calming"

    if "?" in m or any(p in m for p in [
        "is it ok", "is it okay", "can i", "should i", "what", "why", "how",
        "is it normal", "what does it mean", "tell me about", "explain"
    ]):
        return "info_question"

    if any(p in m for p in [
        "pain", "cramp", "bleeding", "spotting", "discharge", "itching", "smell",
        "dizzy", "faint", "vomit", "nausea"
    ]):
        return "symptom"

    return "support"


# -------------------------------
# Topic detector (light)
# -------------------------------
def detect_topic(msg: str) -> str:
    m = msg.lower()

    if any(w in m for w in ["late", "missed period", "no period", "delayed"]):
        return "late_period"
    if any(w in m for w in ["cramp", "cramps", "pain", "hurt", "aching", "back pain"]):
        return "pain_cramps"
    if any(w in m for w in ["heavy bleeding", "soaking", "clots"]):
        return "heavy_bleeding"
    if any(w in m for w in ["spotting", "brown", "light bleeding"]):
        return "spotting"
    if any(w in m for w in ["mood", "irritable", "pms", "sad", "angry"]):
        return "mood_swings"
    if any(w in m for w in ["pad", "tampon", "cup", "hygiene", "wash", "clean"]):
        return "hygiene"
    if any(w in m for w in ["ice cream", "icecream", "coffee", "caffeine", "spicy", "eat", "drink", "diet", "food"]):
        return "food_diet"

    return "unknown"


CALMING_STEPS = [
    "Let us do a quick breathing step: breathe in for 4 seconds, hold for 2, and breathe out for 6. Repeat 3 times.",
    "Try the 5-4-3-2-1 grounding: 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.",
    "Put one hand on your chest and gently relax your shoulders. Tell yourself: \"This feeling will pass. I am safe right now.\"",
]

# -------------------------------
# Safety rules
# -------------------------------
DANGEROUS_SYMPTOMS = [
    "faint",
    "fainted",
    "fever",
    "high fever",
    "severe pain",
    "very heavy bleeding",
    "soaking",
    "large clots",
    "chest pain",
]

MEDICINE_WORDS = [
    "ibuprofen",
    "panadol",
    "paracetamol",
    "mefenamic",
    "mg",
    "dose",
    "dosage",
    "tablet",
]

DIAGNOSIS_PHRASES = [
    "you have",
    "this is pcos",
    "this is endometriosis",
    "you are diagnosed",
]


def apply_safety_constraints(user_text: str, bot_reply: str) -> str:
    """
    Enforce simple, rule-based safety: strip meds/dosage, block diagnosis language,
    and escalate dangerous symptom mentions to trusted adult/clinic.
    """
    u = (user_text or "").lower()
    r = (bot_reply or "").strip()

    # Remove medication/dosage advice
    if any(w in r.lower() for w in MEDICINE_WORDS):
        r = re.sub(r"(?i)\\b(ibuprofen|panadol|paracetamol|mefenamic|mg|dose|dosage|tablet)\\b.*?(\\.|$)", "", r).strip()

    # Prevent diagnosis language
    if any(p in r.lower() for p in DIAGNOSIS_PHRASES):
        r = (
            r
            + "\n\n"
            + "I cannot diagnose what is happening, but a doctor or clinic can help check safely."
        ).strip()

    # Escalate dangerous symptoms
    if any(s in u for s in DANGEROUS_SYMPTOMS):
        r = (
            r
            + "\n\n"
            + "If you have severe pain, fainting, fever, or very heavy bleeding, please tell a trusted adult and visit a clinic as soon as possible."
        ).strip()

    return r


def format_kb_answer(hits) -> str:
    if not hits:
        return ""

    def strip_heading(txt: str) -> str:
        # Remove loud all-caps headings at the start of a chunk.
        return re.sub(r"^[A-Z][A-Z\s]{3,}\s+(?=[A-Z])", "", txt).strip()

    def drop_inline_all_caps(txt: str) -> str:
        # Remove inline ALL-CAPS phrases (e.g., section headers) anywhere in the chunk.
        txt = re.sub(r"\b[A-Z][A-Z\s\(\)\/\-]{3,}\b", "", txt)
        return re.sub(r"\s+", " ", txt).strip()

    parts = []
    for h in hits[:2]:
        cleaned = clean_kb_text(h.chunk, max_sentences=3)
        if cleaned:
            parts.append(drop_inline_all_caps(strip_heading(cleaned)))

    answer = " ".join(parts).strip()
    if len(answer) > 550:
        answer = answer[:550].rsplit(" ", 1)[0] + "..."
    return answer


# -------------------------------
# MAIN ChatService
# -------------------------------
class ChatService:
    def __init__(
        self,
        use_emotions: bool = True,
        use_kb: bool = True,
        kb_backend: str = "embedding",
    ):
        self.use_emotions = use_emotions
        self.use_kb = use_kb
        self.emotion_model = EmotionClassifier() if use_emotions else None
        self.kb = KnowledgeBaseRetriever(
            docs_dir="kb/docs",
            chunk_size=450,
            min_score=0.12 if kb_backend != "embedding" else 0.25,
            backend=kb_backend,
        ) if use_kb else None

    def generate_reply(self, user_message: str) -> ChatResult:
        text = (user_message or "").strip()

        if not text:
            return ChatResult(
                reply="I am here whenever you feel ready to talk.",
                emotions=[],
                raw_emotions=[],
                topic="unknown",
                intent="support",
                kb_sources=[],
            )

        # 1) Emotions (ONLY if enabled)
        raw_emotions = []
        labels: List[str] = []
        emotion_line = ""

        if self.use_emotions and self.emotion_model is not None:
            raw_emotions = self.emotion_model.predict_emotions(text, top_k=3)
            labels = [r.get("label") for r in raw_emotions] if isinstance(raw_emotions, list) else []
            labels = [l for l in labels if l]
            bucket = emotion_bucket(labels)
            emotion_line = choose_emotion_line(bucket)

        # 2) Intent + topic
        intent = detect_intent(text)
        topic = detect_topic(text)

        # 3) KB retrieval (same for both versions)
        kb_hits = []
        kb_sources = []
        kb_answer = ""

        if self.use_kb and self.kb is not None and intent in ["info_question", "symptom"]:
            kb_hits = self.kb.search(text, top_k=2)
            kb_sources = [h.source for h in kb_hits]
            kb_answer = format_kb_answer(kb_hits)

        # helper prefix (only add if emotion is on)
        prefix = (emotion_line + "\n\n") if emotion_line else ""

        # 4) Compose reply
        if intent == "calming":
            calming = random.choice(CALMING_STEPS)
            reply = (
                f"{prefix}"
                f"{calming}\n\n"
                "If you want, tell me what is making you feel worried right now — I am listening."
            )

        elif kb_answer:
            reply = (
                f"{prefix}"
                f"{kb_answer}\n\n"
                "If you have severe pain, fainting, fever, very heavy bleeding, or you feel unsafe, please talk to a trusted adult or visit a clinic."
            )

        elif intent in ["info_question", "symptom"]:
                reply = (
                    f"{prefix}"
                    "I can help! Just to answer accurately: can you tell me a bit more (how long it has been, your age, and any pain, heavy bleeding, fever, or dizziness)?\n\n"
                    "If you feel very unwell or scared, please reach out to a trusted adult or a clinic."
                )
        else:
            # Baseline non-emotion bot should still be polite
            if self.use_emotions:
                reply = (
                    f"{prefix}"
                    "You do not have to handle this alone. If you want, tell me what is happening in your body or what you are worried about."
                )
            else:
                reply = (
                    "I can listen and help. Tell me what is happening in your body or what you want to know."
                )

        reply = apply_safety_constraints(text, reply)

        return ChatResult(
            reply=reply.strip(),
            emotions=labels,
            raw_emotions=raw_emotions,
            topic=topic,
            intent=intent,
            kb_sources=kb_sources,
        )
