from services.chat_service import ChatService

TESTS = [
    "I feel scared because my period is late.",
    "Can I drink coffee when I have cramps?",
    "I feel lonely and I have nobody to talk to.",
    "Is it normal to have smell during periods?",
    "How can I reduce cramps naturally?",
]

bot_with_kb = ChatService(use_kb=True)
bot_without_kb = ChatService(use_kb=False)

for msg in TESTS:
    r_with = bot_with_kb.generate_reply(msg)
    r_without = bot_without_kb.generate_reply(msg)

    print("\n" + "=" * 70)
    print("USER:", msg)
    print("\n--- WITH KB ---")
    print(r_with.reply)
    if r_with.kb_sources:
        print("KB sources:", ", ".join(r_with.kb_sources))
    print("\n--- WITHOUT KB ---")
    print(r_without.reply)
