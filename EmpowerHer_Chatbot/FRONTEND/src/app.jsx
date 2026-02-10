import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

function Bubble({ role, text, meta }) {
  return (
    <div className={`row ${role}`}>
      <div className={`bubble ${role}`}>
        <div className="bubbleText">{text}</div>

        {role === "bot" && meta && (
          <div className="meta">
            <span className="chip">Intent: {meta.intent || "-"}</span>
            <span className="chip">Topic: {meta.topic || "-"}</span>

            <span className="chip">
              Emotions:{" "}
              {Array.isArray(meta.emotions) && meta.emotions.length
                ? meta.emotions.join(", ")
                : "-"}
            </span>

            <span className="chip">
              KB:{" "}
              {Array.isArray(meta.kb_sources) && meta.kb_sources.length
                ? meta.kb_sources.join(", ")
                : "-"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "Hi 💗 I’m EmpowerHer. You can talk to me about your feelings, or ask questions about periods. What’s on your mind?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const endRef = useRef(null);

  const suggestions = useMemo(
    () => [
      "My period is late and I’m scared 😟",
      "How can I reduce cramps naturally?",
      "Is it okay to eat ice cream during periods?",
      "I feel angry and sad before my period",
      "Help me calm down, I’m overthinking",
    ],
    []
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage(text) {
    const msg = (text ?? input).trim();
    if (!msg || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: msg }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      const reply = data?.reply || "Sorry 😕 I couldn’t generate a reply.";

      setMessages((prev) => [
        ...prev,
        { role: "bot", text: reply, meta: data },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: "Oops 😕 I can’t connect to the backend. Is Flask running on http://127.0.0.1:5000 ?",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo">🌸</div>
          <div>
            <h1>EmpowerHer</h1>
            <p>Menstrual Support Chatbot • Emotion + Knowledge Base</p>
          </div>
        </div>

        <div className="topActions">
          <button className="ghost" onClick={() => setMessages(messages.slice(0, 1))}>
            Clear Chat
          </button>
        </div>
      </header>

      <main className="main">
        <section className="chatArea">
          <div className="chatBox">
            {messages.map((m, i) => (
              <Bubble key={i} role={m.role} text={m.text} meta={m.meta} />
            ))}

            {loading && (
              <div className="row bot">
                <div className="bubble bot">
                  <div className="typing">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}

            <div ref={endRef} />
          </div>

          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Type here… (Enter = send, Shift+Enter = new line)"
              rows={2}
            />
            <button className="sendBtn" disabled={loading || !input.trim()} onClick={() => sendMessage()}>
              Send 💌
            </button>
          </div>
        </section>

        <aside className="sidePanel">
          <div className="card">
            <h2>Quick Prompts ✨</h2>
            <p className="muted">Click one to test instantly.</p>

            <div className="suggestions">
              {suggestions.map((s) => (
                <button key={s} className="suggestionBtn" onClick={() => sendMessage(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>Safety Note 🩷</h2>
            <p className="muted">
              I can give general guidance, not medical diagnosis. If symptoms are severe (fainting, heavy bleeding, fever),
              please talk to a trusted adult or a clinic.
            </p>
          </div>
        </aside>
      </main>

      <footer className="footer">
        <span>© EmpowerHer • Local Demo</span>
        <span className="dot">•</span>
        <span>Backend: Flask /chat</span>
      </footer>
    </div>
  );
}
