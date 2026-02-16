import type { KeyboardEventHandler } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

type ChatMeta = {
  intent?: string;
  topic?: string;
  emotions?: string[];
  kb_sources?: string[];
};

type ChatMessage = {
  role: "user" | "bot";
  text: string;
  meta?: ChatMeta;
};

function Bubble({ role, text, meta }: ChatMessage) {
  return (
    <div className={`row ${role}`}>
      <div className={`bubble ${role}`}>
        <div className="bubbleText">{text}</div>

        {role === "bot" && meta && (
          <div className="meta">
            <span className="chip">Intent: {meta.intent ?? "-"}</span>
            <span className="chip">Topic: {meta.topic ?? "-"}</span>
            <span className="chip">
              Emotions:{" "}
              {meta.emotions && meta.emotions.length
                ? meta.emotions.join(", ")
                : "-"}
            </span>
            <span className="chip">
              KB:{" "}
              {meta.kb_sources && meta.kb_sources.length
                ? meta.kb_sources.join(", ")
                : "-"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

const initialBotMessage: ChatMessage = {
  role: "bot",
  text: "Hi, I'm EmpowerHer. I'm here to listen and help with questions about periods, moods, or anything you're feeling. What's on your mind today?",
};

type ChatResponse = {
  reply?: string;
  emotions?: string[];
  raw_emotions?: unknown;
  topic?: string;
  intent?: string;
  kb_sources?: string[];
};

export default function App() {
  const [booting, setBooting] = useState(true);
  const [showLanding, setShowLanding] = useState(true);
  const [messages, setMessages] = useState<ChatMessage[]>([
    initialBotMessage,
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const endRef = useRef<HTMLDivElement | null>(null);

  const suggestions = useMemo(
    () => [
      "My period is late and I'm scared.",
      "How can I reduce cramps naturally?",
      "Is it okay to eat ice cream during periods?",
      "I feel angry and sad before my period.",
      "Help me calm down, I'm overthinking.",
    ],
    []
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const timer = window.setTimeout(() => setBooting(false), 1400);
    return () => window.clearTimeout(timer);
  }, []);

  const sendMessage = async (text?: string) => {
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

      const data: ChatResponse = await res.json();
      const reply =
        data.reply?.trim() ||
        "Sorry, I could not generate a reply right now.";

      const meta: ChatMeta = {
        intent: data.intent ?? undefined,
        topic: data.topic ?? undefined,
        emotions: Array.isArray(data.emotions) ? data.emotions : undefined,
        kb_sources: Array.isArray(data.kb_sources)
          ? data.kb_sources
          : undefined,
      };

      setMessages((prev) => [...prev, { role: "bot", text: reply, meta }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: "Oops, I could not connect to the backend. Is Flask running on http://127.0.0.1:5000 ?",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  if (booting) {
    return (
      <div className="bootScreen">
        <div className="bootCard">
          <div className="bootLogo">
            <img src="/images/Logo.png" alt="EmpowerHer logo" />
          </div>
          <h1>EmpowerHer</h1>
          <p>Preparing your safe space...</p>
          <div className="bootBar">
            <span />
          </div>
          <div className="bootDots">
            <span />
            <span />
            <span />
          </div>
        </div>
      </div>
    );
  }

  if (showLanding) {
    return (
      <div className="landing">
        <header className="landingNav">
          <div className="brand">
            <div className="logo">
              <img src="/images/Logo.png" alt="EmpowerHer logo" />
            </div>
            <div>
              <h1>EmpowerHer</h1>
              <p>Gentle support for periods, moods, and questions.</p>
            </div>
          </div>

          <nav className="landingLinks">
            <button className="navLink">About</button>
            <button className="navLink">Resources</button>
            <button className="navLink">Safety</button>
          </nav>
        </header>

        <main className="landingHero">
          <div className="heroCard">
            <span className="heroTag">Private. Calm. Helpful.</span>
            <h2>EmpowerHer is your safe space for period support.</h2>
            <p>
              Ask questions, track how you feel, and get grounded tips with a
              gentle chatbot designed for teens.
            </p>
            <div className="heroActions">
              <button
                className="primaryBtn"
                onClick={() => setShowLanding(false)}
              >
                Start Chatting
              </button>
              <button className="ghostBtn">Learn More</button>
            </div>
          </div>

          <div className="heroGlow" aria-hidden="true" />
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo">
            <img src="/images/Logo.png" alt="EmpowerHer logo" />
          </div>
          <div>
            <h1>EmpowerHer</h1>
            <p>Gentle, private support for periods, moods, and questions.</p>
          </div>
        </div>

        <div className="topActions">
          <button className="ghost" onClick={() => setShowLanding(true)}>
            Back to Home
          </button>
          <button className="ghost" onClick={() => setMessages([initialBotMessage])}>
            Clear Chat
          </button>
        </div>
      </header>

      <main className="main">
        <section className="chatArea">
          <div className="chatBox">
            {messages.map((m, i) => (
              <Bubble key={`${m.role}-${i}-${m.text.slice(0, 8)}`} role={m.role} text={m.text} meta={m.meta} />
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
              placeholder="Type here (Enter = send, Shift+Enter = new line)"
              rows={2}
            />
            <button
              className="sendBtn"
              disabled={loading || !input.trim()}
              onClick={() => void sendMessage()}
            >
              Send
            </button>
          </div>
        </section>

        <aside className="sidePanel">
          <div className="card">
            <h2>Quick Prompts</h2>
            <p className="muted">Pick one to start a conversation.</p>

            <div className="suggestions">
              {suggestions.map((s) => (
                <button key={s} className="suggestionBtn" onClick={() => void sendMessage(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>Safety Note</h2>
            <p className="muted">
              I can offer general guidance, not medical diagnosis. If symptoms are severe (fainting, heavy bleeding, fever),
              please talk to a trusted adult or healthcare provider.
            </p>
          </div>
        </aside>
      </main>

      <footer className="footer">
        <span>EmpowerHer - Local Demo</span>
        <span className="dot">-</span>
        <span>Backend: Flask /chat</span>
      </footer>
    </div>
  );
}
