# services/chat_service.py

from dataclasses import dataclass
from typing import List, Any, Optional
import random
import re

from models.emotion_classifier import EmotionClassifier
from models.response_generator import EmpatheticResponder
from services.kb_retriever import KnowledgeBaseRetriever, clean_kb_text


@dataclass
class ChatResult:
    reply: str
    emotions: List[str]
    raw_emotions: Any
    topic: str
    intent: str
    kb_sources: List[str]


def _history_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text", "")).strip()
    return ""


def _history_role(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("role", "")).strip().lower()
    return ""


def _history_topic(item: Any) -> str:
    if isinstance(item, dict):
        meta = item.get("meta")
        if isinstance(meta, dict):
            return str(meta.get("topic", "")).strip().lower()
    return ""


def get_recent_context(history: Optional[List[Any]] = None) -> tuple[str, str]:
    recent = history or []
    last_topic = ""
    last_user_text = ""

    for item in reversed(recent):
        role = _history_role(item)
        if not last_topic and role == "bot":
            last_topic = _history_topic(item)
        if not last_user_text and role == "user":
            last_user_text = _history_text(item)
        if last_topic and last_user_text:
            break

    return last_topic, last_user_text


FOLLOW_UP_HINTS = {
    "late_period": [
        "late period", "missed period", "period late", "late", "delay", "delayed period",
    ],
    "pain_cramps": [
        "cramps", "cramp pain", "pain relief", "reduce cramps", "period pain",
    ],
    "food_diet": [
        "food", "foods", "diet", "drink", "drinks", "eat", "eating", "ice cream", "coffee",
    ],
    "mood_swings": [
        "mood", "sad", "angry", "irritable", "before my period",
    ],
    "hygiene": [
        "hygiene", "wash", "clean", "pad change", "bath", "shower",
    ],
    "bathing_swimming": [
        "swim", "swimming", "bath", "bathing", "shower", "pool",
    ],
    "first_period": [
        "first period", "menarche", "started my period", "got my first period",
    ],
    "normal_discharge": [
        "discharge", "white discharge", "clear discharge", "yellow discharge",
    ],
}


FOLLOW_UP_SIGNAL_PATTERNS = [
    r"\b\d+\s*(day|days|week|weeks|month|months)\b",
    r"\band\b",
    r"\bbut\b",
    r"\balso\b",
    r"\bjust\b",
    r"\bstill\b",
    r"\bonly\b",
    r"\bfever\b",
    r"\bdizzy\b",
    r"\bfaint(ing|ed)?\b",
    r"\bbleeding\b",
    r"\bheavy\b",
]


def _looks_like_follow_up_fragment(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    if "?" in low:
        return False
    if any(low.startswith(prefix) for prefix in ["what ", "why ", "how ", "can ", "is ", "should ", "when "]):
        return False
    if len(low.split()) <= 5:
        return True
    return any(re.search(pattern, low) for pattern in FOLLOW_UP_SIGNAL_PATTERNS)


def enrich_follow_up_message(user_message: str, history: Optional[List[Any]] = None) -> str:
    text = (user_message or "").strip()
    if not text:
        return text

    last_topic, last_user_text = get_recent_context(history)

    if not last_topic:
        return text

    hints = FOLLOW_UP_HINTS.get(last_topic, [])
    if any(h in text.lower() for h in hints):
        return text

    if not _looks_like_follow_up_fragment(text):
        return text

    if last_user_text:
        return f"{text} (follow-up to: {last_user_text}; topic: {last_topic})"
    return f"{text} (follow-up topic: {last_topic})"


def is_follow_up_message(user_message: str, history: Optional[List[Any]] = None) -> bool:
    text = (user_message or "").strip()
    if not text:
        return False
    last_topic, _ = get_recent_context(history)
    if not last_topic:
        return False
    hints = FOLLOW_UP_HINTS.get(last_topic, [])
    if any(h in text.lower() for h in hints):
        return False
    return _looks_like_follow_up_fragment(text)


def extract_duration_phrase(text: str) -> str:
    match = re.search(r"\b\d+\s*(day|days|week|weeks|month|months)\b", (text or "").lower())
    return match.group(0) if match else ""


def build_follow_up_reply(current_text: str, previous_topic: str, kb_answer: str = "") -> str:
    low = (current_text or "").lower()
    duration = extract_duration_phrase(current_text)
    dangerous = has_dangerous_symptoms(current_text)

    if previous_topic == "late_period":
        has_concerning_detail = dangerous or any(p in low for p in ["dizzy", "dizziness", "very unwell", "weak"])
        if has_concerning_detail:
            detail = (
                f" Being late for {duration} with these symptoms needs medical advice."
                if duration else
                ""
            )
            return (
                f"A late period with symptoms like fever, fainting, severe pain, dizziness, or very heavy bleeding is not something to ignore.{detail} "
                "Please tell a trusted adult and get medical advice from a clinic or doctor as soon as possible."
            )
        if duration:
            return (
                f"A delay of {duration} can sometimes happen because of stress, sleep changes, illness, travel, or normal hormone shifts. "
                "Please keep watching for pain, heavy bleeding, dizziness, or fever, and tell me if any of those are happening."
            )

    if previous_topic == "pain_cramps":
        if dangerous or any(p in low for p in ["cannot stand", "can't stand", "cant stand", "can't sleep", "cant sleep", "school"]):
            return (
                "If cramps are so strong that you cannot sleep, stand, or do normal activities, please tell a trusted adult and get medical advice. "
                "Severe pain should not be ignored."
            )
        if any(p in low for p in ["heat", "warm", "water", "rest", "stretch", "walk", "natural"]):
            return (
                "Gentle heat, rest, drinking water, light stretching, and slow walking can help some girls with cramps. "
                "If the pain keeps getting worse or lasts many days, please talk to a trusted adult or a doctor."
            )

    if previous_topic in ["odor_smell", "normal_discharge"]:
        if dangerous or any(p in low for p in ["itch", "itching", "burn", "burning", "green", "yellow", "fishy"]):
            return (
                "A strong smell or unusual discharge with itching, burning, pain, fever, or a green or yellow color should be checked. "
                "Please tell a trusted adult or visit a clinic."
            )

    return build_kb_reply(kb_answer) if kb_answer else ""


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

    if m in ["any other", "any others", "anything else", "other methods", "other techniques"]:
        return "calming"

    if any(p in m for p in [
        "i love", "i am happy", "i'm happy", "i feel better", "feels better",
        "that helps", "this helps", "i appreciate", "i am grateful", "i'm grateful",
        "thank you", "thanks"
    ]):
        return "affirmation"

    if any(p in m for p in [
        "late period", "missed period", "missed my period", "no period",
        "didnt get my period", "didn't get my period", "did not get my period",
        "not got my period", "period is late", "period late", "period delay"
    ]):
        return "info_question"

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

    if any(w in m for w in ["family", "mom", "mother", "dad", "father", "parent", "sister", "brother"]):
        return "family_support"

    if any(w in m for w in [
        "when should i see a doctor", "see a doctor", "see a nurse", "clinic",
        "when to see a doctor", "when should i go", "need a doctor"
    ]):
        return "doctor_when"

    if any(w in m for w in [
        "first period", "my first period", "got my first period", "start my period",
        "started my period", "menarche", "when will my first period come",
    ]):
        return "first_period"

    if any(w in m for w in [
        "swim", "swimming", "pool", "bath during my period", "bathing during my period",
        "shower during my period", "bathe during my period", "take a bath", "taking a bath",
        "take a shower", "taking a shower",
    ]):
        return "bathing_swimming"

    if any(w in m for w in ["cup", "menstrual cup"]):
        return "menstrual_cup"
    if any(w in m for w in ["tampon"]):
        return "tampon"
    if any(w in m for w in ["pad", "pads"]):
        return "pads"
    if any(w in m for w in ["smell", "odor", "odour", "fishy"]):
        return "odor_smell"
    if any(w in m for w in ["coffee", "caffeine"]):
        return "caffeine"
    if any(w in m for w in ["exercise", "workout", "gym", "sports", "pe class"]):
        return "exercise"
    if any(w in m for w in ["normal cycle length", "cycle length for teens", "cycle length", "irregular", "calendar", "app", "track my cycle", "track my period"]):
        return "cycle_tracking"
    if any(w in m for w in [
        "bloating", "food", "diet", "eat", "drink", "water", "dehydration",
        "dehydrated", "ice cream", "ice-cream", "chocolate", "snack", "snacks"
    ]):
        return "food_diet"
    if any(w in m for w in ["nausea", "vomit", "diarrhea", "diarrhoea"]):
        return "stomach"
    if any(w in m for w in ["breast", "sore", "soreness", "tender"]):
        return "breast_soreness"
    if any(w in m for w in ["dizzy", "dizziness", "faint", "fainting", "shaky", "lightheaded"]):
        return "dizziness"
    if any(w in m for w in ["sleep", "night", "bed"]):
        return "sleep"
    if any(w in m for w in ["leak", "leaking", "stain", "stained", "bleed through", "soak through"]):
        return "leaking"

    if any(w in m for w in [
        "late", "missed period", "missed my period", "no period", "delayed",
        "didnt get my period", "didn't get my period", "did not get my period",
        "not got my period", "period is late", "period late", "period delay"
    ]):
        return "late_period"
    if any(w in m for w in ["cramp", "cramps", "pain", "hurt", "aching", "back pain"]):
        return "pain_cramps"
    if any(w in m for w in ["heavy bleeding", "soaking", "clots"]):
        return "heavy_bleeding"
    if any(w in m for w in ["spotting", "brown", "light bleeding"]):
        return "spotting"
    if any(w in m for w in [
        "discharge", "white discharge", "clear discharge", "yellow discharge",
        "green discharge", "vaginal discharge",
    ]):
        return "normal_discharge"
    if any(w in m for w in ["mood", "irritable", "pms", "sad", "angry"]):
        return "mood_swings"
    if any(w in m for w in ["hygiene", "wash", "clean", "shower", "bath"]):
        return "hygiene"

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
    "medicine",
    "medication",
    "drug",
    "mg",
    "dose",
    "dosage",
    "tablet",
]

DIAGNOSIS_PHRASES = [
    "this is pcos",
    "this is endometriosis",
    "you are diagnosed",
    "you definitely have",
    "it sounds like you have",
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
    if any(s in u for s in DANGEROUS_SYMPTOMS) and not (
        "trusted adult" in r.lower() and ("clinic" in r.lower() or "doctor" in r.lower())
    ):
        r = (
            r
            + "\n\n"
            + "If you have severe pain, fainting, fever, or very heavy bleeding, please tell a trusted adult and visit a clinic as soon as possible."
        ).strip()

    return r


def has_dangerous_symptoms(text: str) -> bool:
    low = (text or "").lower()
    return any(s in low for s in DANGEROUS_SYMPTOMS)


def trim_to_sentences(text: str, max_sentences: int = 4) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()


GENERIC_BANNED_PHRASES = [
    "as an ai",
    "i am an ai",
    "i am a bot",
    "as a chatbot",
    "language model",
    "empowerher",
]


def cleanup_reply(text: str, max_sentences: int = 4) -> str:
    if not text:
        return text
    r = text.strip()
    low = r.lower()
    for p in GENERIC_BANNED_PHRASES:
        if p in low:
            r = re.sub(r"(?i).*" + re.escape(p) + r".*?(\.|$)", "", r).strip()
            low = r.lower()
    # Remove any remaining quotes that look like "EmpowerHer says ..."
    r = re.sub(r'(?i)^"?\s*empowerher\s+(says|said)\s*[:,]?\s*"?', "", r).strip()
    r = re.sub(r"\s+\n", "\n", r).strip()
    r = trim_to_sentences(r, max_sentences=max_sentences)
    r = re.sub(r"^[\W_]+$", "", r).strip()
    return r.strip()


TOPIC_TEMPLATES = {
    "late_period": (
        "It can feel worrying when your period is late, and you are not alone in that. "
        "Stress, changes in routine, weight changes, and hormones can all shift timing. "
        "How long has it been late, and are you having pain, heavy bleeding, dizziness, or fever?"
    ),
    "pain_cramps": (
        "Cramps can be really painful, and it makes sense that you feel upset. "
        "Gentle heat, rest, and drinking water can sometimes help a little. "
        "If pain is very strong, lasts many days, or stops you from normal activities, please talk to a trusted adult or a doctor."
    ),
    "heavy_bleeding": (
        "Heavy bleeding can feel scary, and it is okay to ask for help. "
        "If you are soaking pads very quickly, passing large clots, or feel dizzy or weak, please talk to a trusted adult or go to a clinic."
    ),
    "spotting": (
        "Light spotting can happen for different reasons and can be normal for some girls. "
        "If the bleeding becomes heavy, lasts many days, or comes with strong pain or fever, please reach out to a trusted adult or a clinic."
    ),
    "mood_swings": (
        "Mood changes before a period are common, and your feelings are valid. "
        "It may help to rest, write your feelings down, or talk to someone you trust. "
        "If these mood changes feel too strong or last a long time, a doctor or counsellor can help."
    ),
    "hygiene": (
        "It is good to keep the area clean and dry using gentle, unscented products. "
        "Change pads or tampons regularly, and wash hands before and after. "
        "If you have itching, pain, or a strong smell that does not go away, please ask a trusted adult or a clinic for help."
    ),
    "food_diet": (
        "Food and drink can sometimes affect how you feel during your period. "
        "Water, warm drinks, and balanced meals can help you feel steadier. "
        "If you want, tell me what you have been eating and how your body feels."
    ),
    "bathing_swimming": (
        "Yes, bathing during your period is safe and can help you feel clean and comfortable. "
        "Some girls also swim during their period using suitable menstrual products, while others prefer not to during heavy flow. "
        "If you feel pain, dizziness, or strong discomfort, please stop and tell a trusted adult."
    ),
    "first_period": (
        "The first period often starts between ages 9 and 15, and it is common to feel unsure about it. "
        "Signs before a first period can include white discharge, breast development, body hair growth, and mood changes. "
        "Keeping pads ready and talking to a trusted adult can help you feel more prepared."
    ),
    "menstrual_cup": (
        "Menstrual cups can be safe when cleaned well. "
        "Wash your hands, rinse with clean water, and follow the cup instructions for cleaning. "
        "If you have pain or irritation, take a break and talk to a trusted adult or a clinic."
    ),
    "tampon": (
        "It is okay to use a tampon if you feel ready, and many girls do. "
        "Change it regularly and follow the instructions in the box. "
        "If you feel pain, remove it and ask a trusted adult or a clinic for help."
    ),
    "pads": (
        "It helps to change pads regularly so you stay clean and comfortable. "
        "If you are bleeding heavily, you may need to change more often. "
        "If you are soaking pads very quickly or feel dizzy, please tell a trusted adult."
    ),
    "odor_smell": (
        "A mild smell can be normal, but a strong or fishy smell can feel worrying. "
        "Try gentle hygiene and change pads or tampons regularly. "
        "If the smell is strong, continues, or comes with itching or pain, please talk to a trusted adult or a clinic."
    ),
    "normal_discharge": (
        "Clear or white discharge with little or no smell can be normal and helps keep the vagina clean and healthy. "
        "Please seek help if the discharge is green or yellow, smells strong, or comes with itching, burning, or pain. "
        "Gentle washing with water and avoiding harsh soaps can help protect the area."
    ),
    "caffeine": (
        "Some people find caffeine can make cramps or anxiety feel stronger. "
        "If you notice more pain after coffee or soda, try cutting back and see how your body feels. "
        "Warm water or herbal tea can be a gentler choice."
    ),
    "exercise": (
        "Light exercise can be okay during your period if you feel up to it. "
        "Gentle stretching, walking, or slow movement can sometimes ease cramps. "
        "If pain is strong or you feel dizzy, rest and tell a trusted adult."
    ),
    "cycle_tracking": (
        "Tracking your cycle can help you feel more in control. "
        "You can mark the first day of your period each month on a calendar or app. "
        "If your cycles are very irregular or you are worried, a doctor or nurse can help."
    ),
    "stomach": (
        "Some girls feel nausea or changes in their stomach during periods, and it can be uncomfortable. "
        "Small meals, warm drinks, and rest can sometimes help. "
        "If you have severe vomiting, diarrhea, or feel very unwell, please talk to a trusted adult or a clinic."
    ),
    "breast_soreness": (
        "Breast soreness before a period can be common and usually settles after bleeding starts. "
        "A supportive bra and gentle rest can help. "
        "If the pain is sharp, one-sided, or very strong, please talk to a trusted adult or a clinic."
    ),
    "dizziness": (
        "Feeling dizzy can be scary. "
        "Try to sit or lie down, drink water, and eat something light if you can. "
        "If you faint, have very heavy bleeding, or feel very weak, please tell a trusted adult and seek medical help."
    ),
    "sleep": (
        "Sleep can be harder during cramps. "
        "Gentle heat, a comfortable position, and slow breathing can help you relax. "
        "If pain keeps waking you up often, please talk to a trusted adult or a doctor."
    ),
    "leaking": (
        "Leaking is stressful, and you are not alone in that. "
        "You can use a longer pad at night, change before bed, and keep a spare pad with you. "
        "If bleeding is very heavy or you soak through quickly, please tell a trusted adult."
    ),
    "doctor_when": (
        "It can be hard to know when to see a doctor, and it is okay to ask. "
        "Please seek help if you have very strong pain, fainting, fever, very heavy bleeding, or pain that stops you from normal activities. "
        "If you are unsure, a nurse or doctor can help you decide what is safest."
    ),
    "family_support": (
        "It is really meaningful to feel supported by your family. "
        "You deserve that kind of care and kindness. "
        "If you want to share more about how you are feeling, I am here to listen."
    ),
}


def build_specific_topic_reply(user_text: str, topic: str) -> str:
    text = (user_text or "").lower()

    if topic == "food_diet":
        if "ice cream" in text or "ice-cream" in text:
            return (
                "Yes, eating ice cream during your period is usually okay. "
                "Cold foods do not stop your period, but some girls feel they make cramps feel worse. "
                "If you notice that happens to you, have a smaller amount or choose something warmer instead."
            )
        if "coffee" in text or "caffeine" in text:
            return (
                "Coffee is not forbidden during your period, but caffeine can make cramps, anxiety, or bloating feel worse for some people. "
                "If you notice that, try cutting back and drinking more water instead."
            )

    return TOPIC_TEMPLATES.get(topic, "")


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


def build_rag_context(hits) -> str:
    if not hits:
        return ""

    context_parts = []
    for idx, hit in enumerate(hits[:3], start=1):
        chunk = clean_kb_text(hit.chunk, max_sentences=4)
        if chunk:
            context_parts.append(f"[Source {idx}: {hit.source}]\n{chunk}")

    return "\n\n".join(context_parts).strip()


def should_answer_from_kb(intent: str, topic: str, hits) -> bool:
    if not hits:
        return False
    top_score = float(hits[0].score)
    if intent in ["info_question", "symptom"]:
        return top_score >= 0.22
    if topic == "unknown":
        return top_score >= 0.45
    return False


def build_kb_reply(answer: str) -> str:
    if not answer:
        return ""
    return (
        f"{answer}\n\n"
        "If you have severe pain, fainting, fever, very heavy bleeding, or you feel unsafe, please talk to a trusted adult or visit a clinic."
    )


# -------------------------------
# MAIN ChatService
# -------------------------------
class ChatService:
    def __init__(
        self,
        use_emotions: bool = True,
        use_kb: bool = True,
        kb_backend: str = "embedding",
        use_llm: bool = True,
        use_rag: bool = True,
    ):
        self.use_emotions = use_emotions
        self.use_kb = use_kb
        self.emotion_model = EmotionClassifier() if use_emotions else None
        self.use_llm = use_llm
        self.use_rag = use_rag
        self.responder = EmpatheticResponder() if use_llm else None
        self.kb = KnowledgeBaseRetriever(
            docs_dir="kb/docs",
            chunk_size=450,
            min_score=0.12 if kb_backend != "embedding" else 0.25,
            backend=kb_backend,
        ) if use_kb else None

    def generate_reply(self, user_message: str, history: Optional[List[Any]] = None) -> ChatResult:
        original_text = (user_message or "").strip()
        follow_up = is_follow_up_message(original_text, history=history)
        previous_topic, _ = get_recent_context(history)
        text = enrich_follow_up_message(original_text, history=history)

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
        detected_topic = detect_topic(text)
        topic = detected_topic
        # Only inherit the previous topic when the new message is genuinely vague.
        # If we can detect a concrete new topic, keep it instead of forcing follow-up context.
        if follow_up and previous_topic and detected_topic in ["unknown", previous_topic]:
            topic = previous_topic
            if intent == "support":
                intent = "info_question"
        has_topic_template = topic in TOPIC_TEMPLATES

        # 3) KB retrieval (same for both versions)
        kb_hits = []
        kb_sources = []
        kb_answer = ""
        rag_context = ""

        if self.use_kb and self.kb is not None:
            kb_hits = self.kb.search(text, top_k=3)
            kb_sources = [h.source for h in kb_hits]
            kb_answer = format_kb_answer(kb_hits)
            rag_context = build_rag_context(kb_hits)

        # Only add emotional prefacing for supportive or symptom-heavy cases.
        use_prefix = bool(emotion_line) and intent in ["support", "affirmation", "symptom", "calming"]
        prefix = (emotion_line + "\n\n") if use_prefix else ""
        follow_up_reply = build_follow_up_reply(original_text, previous_topic, kb_answer)

        # 4) Compose reply
        if intent == "calming":
            calming = random.choice(CALMING_STEPS)
            reply = (
                f"{prefix}"
                f"{calming}\n\n"
                "If you want, tell me what is making you feel worried right now — I am listening."
            )

        elif follow_up and previous_topic and follow_up_reply:
            reply = f"{prefix}{follow_up_reply}"

        elif has_topic_template:
            reply = f"{prefix}{build_specific_topic_reply(text, topic)}"

        elif kb_answer and should_answer_from_kb(intent, topic, kb_hits):
            reply = f"{prefix}{build_kb_reply(kb_answer)}"

        elif self.use_llm and self.responder is not None and intent in ["support", "affirmation"]:
            llm_reply = self.responder.generate(text, labels or None)
            llm_reply = cleanup_reply(llm_reply, max_sentences=4)
            reply = llm_reply or (
                f"{prefix}"
                "You do not have to handle this alone. If you want, tell me what is happening in your body or what you are worried about."
            )

        elif self.use_llm and self.responder is not None and topic == "unknown":
            llm_reply = self.responder.generate(text, labels or None)
            llm_reply = cleanup_reply(llm_reply, max_sentences=4)
            reply = llm_reply or (
                f"{prefix}"
                "You do not have to handle this alone. If you want, tell me what is happening in your body or what you are worried about."
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

        reply = apply_safety_constraints(original_text, reply)
        max_sentences = 6 if has_dangerous_symptoms(original_text) else 4
        reply = cleanup_reply(reply, max_sentences=max_sentences)
        if not reply:
            reply = "I am here with you. If you want, tell me a little more about what you are feeling."

        return ChatResult(
            reply=reply.strip(),
            emotions=labels,
            raw_emotions=raw_emotions,
            topic=topic,
            intent=intent,
            kb_sources=kb_sources,
        )
