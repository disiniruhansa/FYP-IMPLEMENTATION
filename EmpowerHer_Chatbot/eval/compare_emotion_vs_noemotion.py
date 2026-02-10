from services.chat_service import ChatService

TESTS = [
    "I feel scared because my period is late.",
    "Can I drink coffee when I have cramps?",
    "I feel lonely and I have nobody to talk to.",
    "Is it normal to have smell during periods?",
    "How can I reduce cramps naturally?",
]

bot_emotion = ChatService(use_emotions=True)
bot_no_emotion = ChatService(use_emotions=False)

for msg in TESTS:
    r1 = bot_emotion.generate_reply(msg).reply
    r2 = bot_no_emotion.generate_reply(msg).reply

    print("\n" + "="*70)
    print("USER:", msg)
    print("\n--- WITH EMOTION ---")
    print(r1)
    print("\n--- WITHOUT EMOTION ---")
    print(r2)
