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
  const [showLearnMore, setShowLearnMore] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    initialBotMessage,
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const endRef = useRef<HTMLDivElement | null>(null);
  const learnMoreRef = useRef<HTMLDivElement | null>(null);

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

  useEffect(() => {
    if (showLearnMore) {
      learnMoreRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [showLearnMore]);

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
        body: JSON.stringify({
          message: msg,
          history: messages.slice(-6),
        }),
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
              <button
                className="ghostBtn"
                onClick={() => setShowLearnMore((prev) => !prev)}
              >
                {showLearnMore ? "Hide Details" : "Learn More"}
              </button>
            </div>
          </div>

          <div className="heroGlow" aria-hidden="true" />
        </main>

        {showLearnMore && (
          <section className="learnMore" ref={learnMoreRef}>
            <div className="learnMoreCard">
              <h2>Learn More About EmpowerHer </h2>
              <p>
                EmpowerHer is an emotionally sensitive menstrual health chatbot created to
                support adolescents who want a private, judgment-free way to learn about
                periods and talk about how they feel. Menstrual health is not only about
                cycle dates and symptoms — it can also include emotions like anxiety,
                embarrassment, stress, confusion, or loneliness. Many existing period apps
                are great at tracking, but they often feel clinical and don’t always help
                users feel understood. EmpowerHer was designed to fill that gap by combining
                emotional support and verified menstrual health education in one simple chat
                experience.
              </p>

              <h3>Why EmpowerHer Exists</h3>
              <p>
                For many adolescents, menstruation can feel difficult to talk about openly
                due to social stigma, cultural taboos, or fear of being judged. Some people
                don’t feel comfortable asking a parent, teacher, or friend — especially when
                the question feels too personal. EmpowerHer aims to create a safe digital
                space where users can ask questions, express emotions, and receive calm,
                supportive replies while also getting accurate information.
              </p>

              <h3>What EmpowerHer Can Help With</h3>
              <p>EmpowerHer supports two major types of conversations.</p>

              <div className="learnMoreBlock">
                <h4>1) Menstrual Health Questions (Information Support)</h4>
                <p>You can ask questions such as:</p>
                <ul>
                  <li>Why do cramps happen and what can I do to manage them?</li>
                  <li>Is it normal for my cycle to change sometimes?</li>
                  <li>What should I do if I feel tired or bloated during my period?</li>
                  <li>How can I maintain good hygiene during menstruation?</li>
                  <li>What foods and habits may help with comfort?</li>
                </ul>
                <p>
                  EmpowerHer responds using a verified knowledge base, meaning it focuses
                  on reliable general guidance rather than guessing or making up facts.
                </p>
              </div>

              <div className="learnMoreBlock">
                <h4>2) Emotional Support (Feeling Understood)</h4>
                <p>Sometimes you don’t just want facts — you want reassurance:</p>
                <ul>
                  <li>“I feel nervous about my period.”</li>
                  <li>“I’m embarrassed to talk about this.”</li>
                  <li>“I feel stressed because my period is late.”</li>
                  <li>“I feel uncomfortable and I don’t know what to do.”</li>
                </ul>
                <p>
                  The chatbot is designed to recognize emotional cues and respond in a warm,
                  empathetic tone, helping you feel supported and less alone.
                </p>
              </div>

              <h3>How EmpowerHer Works (Simple Explanation)</h3>
              <ol>
                <li>It detects emotional signals in your text (for example, anxiety or sadness).</li>
                <li>It decides whether you’re mainly asking for information or emotional support.</li>
                <li>If you ask a health question, it retrieves safe, verified guidance from the knowledge base.</li>
                <li>It generates a supportive response in friendly language.</li>
                <li>Finally, it applies safety rules to ensure the response is appropriate and responsible.</li>
              </ol>
              <p>
                This approach helps EmpowerHer stay both kind and safe — especially important
                for adolescent users.
              </p>

              <h3>Safety and Privacy Commitment</h3>
              <div className="learnMoreGrid">
                <div>
                  <h4>No diagnosis or treatment</h4>
                  <p>
                    EmpowerHer does not diagnose medical conditions, prescribe treatment, or
                    recommend medication. It provides general educational support only.
                  </p>
                </div>
                <div>
                  <h4>Supportive and non-judgmental tone</h4>
                  <p>
                    The chatbot avoids shaming language and is designed to be respectful and
                    culturally sensitive.
                  </p>
                </div>
                <div>
                  <h4>Privacy-aware design</h4>
                  <p>
                    EmpowerHer is built to avoid collecting personal identifying details. You
                    should also avoid sharing personal information like your full name,
                    address, school, or phone number inside the chat.
                  </p>
                </div>
              </div>

              <h3>When You Should Seek Real Help</h3>
              <p>
                EmpowerHer is helpful for education and emotional reassurance, but it is not
                a replacement for real medical care. If you experience any of the following,
                please talk to a trusted adult and seek professional support:
              </p>
              <ul>
                <li>Very severe pain that doesn’t improve</li>
                <li>Heavy bleeding that feels unusual</li>
                <li>Fainting, dizziness, or extreme weakness</li>
                <li>Symptoms that cause serious worry</li>
                <li>Any situation where you feel unsafe or overwhelmed</li>
              </ul>

              <h3>Who EmpowerHer Is For</h3>
              <ul>
                <li>Adolescents and young users who want a private place to learn and talk</li>
                <li>Users who want both emotional support and menstrual health guidance</li>
                <li>Anyone who prefers simple explanations rather than complex medical language</li>
              </ul>

              <h3>The Goal</h3>
              <p>
                EmpowerHer’s goal is simple: Help users feel informed, supported, and confident
                when dealing with menstruation — without fear or judgment.
              </p>
              <p className="learnMoreCta">
                If you’re ready, you can start chatting now. You can ask a question, describe
                how you feel, or just say “Hi” — EmpowerHer will respond gently and guide you
                step by step.
              </p>
            </div>
          </section>
        )}
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
