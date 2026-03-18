from services.chat_service import ChatService


TESTS = [
    "I feel scared because my period is late.",
    "Can I drink coffee when I have cramps?",
    "Is it normal to have smell during periods?",
    "How can I reduce cramps naturally?",
    "I have brown spotting and I am worried. What should I do?",
]


bot_with_rag = ChatService(use_kb=True, use_rag=True)
bot_without_rag = ChatService(use_kb=True, use_rag=False)


for msg in TESTS:
    r_rag = bot_with_rag.generate_reply(msg)
    r_nonrag = bot_without_rag.generate_reply(msg)

    print("\n" + "=" * 70)
    print("USER:", msg)

    print("\n--- WITH RAG ---")
    print(r_rag.reply)
    if r_rag.kb_sources:
        print("KB sources:", ", ".join(r_rag.kb_sources))

    print("\n--- WITHOUT RAG ---")
    print(r_nonrag.reply)
    if r_nonrag.kb_sources:
        print("KB sources:", ", ".join(r_nonrag.kb_sources))
