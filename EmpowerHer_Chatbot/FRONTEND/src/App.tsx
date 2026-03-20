import type { KeyboardEventHandler } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import appLogo from "../images/Logo.png";

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

type ChatResponse = {
  reply?: string;
  emotions?: string[];
  raw_emotions?: unknown;
  topic?: string;
  intent?: string;
  kb_sources?: string[];
};

type SectionKey = "home" | "about" | "features" | "why" | "support";

const initialBotMessage: ChatMessage = {
  role: "bot",
  text: "Hi, I'm EmpowerHer. I'm here to listen and help with questions about periods, moods, or anything you're feeling. What's on your mind today?",
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

export default function App() {
  const [booting, setBooting] = useState(true);
  const [showLanding, setShowLanding] = useState(true);
  const [chatView, setChatView] = useState<"chat" | "medical">("chat");
  const [messages, setMessages] = useState<ChatMessage[]>([initialBotMessage]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const endRef = useRef<HTMLDivElement | null>(null);
  const sectionRefs: Record<SectionKey, React.RefObject<HTMLDivElement | null>> =
    {
      home: useRef<HTMLDivElement | null>(null),
      about: useRef<HTMLDivElement | null>(null),
      features: useRef<HTMLDivElement | null>(null),
      why: useRef<HTMLDivElement | null>(null),
      support: useRef<HTMLDivElement | null>(null),
    };

  const suggestions = useMemo(
    () => [
      "My period is late and I'm scared.",
      "How can I reduce cramps naturally?",
      "Is it okay to eat ice cream during periods?",
      "Why does my period smell fishy?",
      "Help me calm down, I'm overthinking.",
    ],
    []
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const timer = window.setTimeout(() => setBooting(false), 1200);
    return () => window.clearTimeout(timer);
  }, []);

  const scrollToSection = (key: SectionKey) => {
    sectionRefs[key].current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

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
    } catch {
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
            <img src={appLogo} alt="EmpowerHer logo" />
          </div>
          <h1>EmpowerHer</h1>
          <p>Preparing your safe support space...</p>
          <div className="bootBar">
            <span />
          </div>
        </div>
      </div>
    );
  }

  if (showLanding) {
    return (
      <div className="siteShell">
        <header className="siteNav">
          <button className="siteBrand" onClick={() => scrollToSection("home")}>
            EmpowerHer
          </button>
          <nav className="siteLinks">
            <button onClick={() => scrollToSection("home")}>Home</button>
            <button onClick={() => scrollToSection("about")}>About</button>
            <button onClick={() => scrollToSection("features")}>Features</button>
            <button onClick={() => scrollToSection("why")}>Why EmpowerHer</button>
            <button className="navCta" onClick={() => scrollToSection("support")}>
              Get Support
            </button>
          </nav>
        </header>

        <section className="heroSection" ref={sectionRefs.home}>
          <div className="heroOverlay" />
          <div className="heroContent">
            <p className="heroLead">A safe and supportive</p>
            <h1>Menstrual Health Chatbot</h1>
            <h2>for Adolescents</h2>
            <p className="heroSubtext">
              Private menstrual health support with emotionally supportive responses,
              clear guidance, and a safe space to ask sensitive questions.
            </p>
            <div className="heroActions">
              <button className="primaryBtn" onClick={() => setShowLanding(false)}>
                Explore the Chatbot
              </button>
              <button className="secondaryBtn" onClick={() => scrollToSection("support")}>
                Learn More
              </button>
            </div>
          </div>
        </section>

        <section className="introBand" ref={sectionRefs.about}>
          <div className="container narrow">
            <p>
              EmpowerHer provides private, emotionally supportive, and easy-to-understand
              menstrual health support for adolescents and young women seeking safe digital guidance.
            </p>
          </div>
        </section>

        <section className="contentSection">
          <div className="container">
            <div className="sectionHeading">
              <h2>Why EmpowerHer Matters</h2>
              <p>
                EmpowerHer was created for adolescents who may feel shy, anxious, or unsupported
                when seeking menstrual health information.
              </p>
            </div>

            <div className="featureGrid">
              <article className="featureCard">
                <div className="featureIcon">🔒</div>
                <h3>Private and Safe</h3>
                <p>
                  A judgment-free place where users can ask sensitive questions without worrying
                  about privacy or stigma.
                </p>
              </article>
              <article className="featureCard">
                <div className="featureIcon">💜</div>
                <h3>Emotionally Supportive</h3>
                <p>
                  Responses are designed to acknowledge fear, stress, confusion, and other emotional
                  needs around menstruation.
                </p>
              </article>
              <article className="featureCard">
                <div className="featureIcon">📘</div>
                <h3>Educational</h3>
                <p>
                  Reliable, age-appropriate menstrual health guidance using a curated local knowledge base.
                </p>
              </article>
            </div>
          </div>
        </section>

        <section className="splitSection" ref={sectionRefs.features}>
          <div className="container splitLayout">
            <div className="imagePanel">
              <img
                src="/images/Celebrating sisterhood in soft pastels.png"
                alt="EmpowerHer support illustration"
              />
            </div>
            <div className="textPanel">
              <h2>What EmpowerHer Offers</h2>
              <ul className="offerList">
                <li>
                  <strong>24/7 chatbot support</strong>
                  <span>Ask menstrual health questions anytime in a private space.</span>
                </li>
                <li>
                  <strong>Emotional reassurance</strong>
                  <span>Get supportive responses that validate feelings and reduce fear.</span>
                </li>
                <li>
                  <strong>Health education</strong>
                  <span>Learn about periods, symptoms, hygiene, food, and warning signs.</span>
                </li>
                <li>
                  <strong>Confidence building</strong>
                  <span>Understand your body with simple explanations designed for adolescents.</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        <section className="lavenderSection" ref={sectionRefs.why}>
          <div className="container">
            <div className="sectionHeading">
              <h2>Who It Is For</h2>
              <p>
                EmpowerHer is designed for young users seeking trustworthy menstrual health support.
              </p>
            </div>
            <div className="miniGrid">
              <article className="miniCard">
                <div className="miniIcon">👩</div>
                <h3>Adolescents</h3>
                <p>Young women starting their menstrual health journey.</p>
              </article>
              <article className="miniCard">
                <div className="miniIcon">🛡️</div>
                <h3>Privacy Seekers</h3>
                <p>Users who prefer anonymous and supportive health conversations.</p>
              </article>
              <article className="miniCard">
                <div className="miniIcon">📝</div>
                <h3>Learners</h3>
                <p>Anyone who wants reliable menstrual health information in simple language.</p>
              </article>
              <article className="miniCard">
                <div className="miniIcon">💬</div>
                <h3>Support Seekers</h3>
                <p>Users who want emotional reassurance alongside health guidance.</p>
              </article>
            </div>
          </div>
        </section>

        <section className="contentSection" ref={sectionRefs.support}>
          <div className="container supportGrid">
            <div className="signupCard">
              <h2>Get Support</h2>
              <p>
                Start chatting with EmpowerHer for private menstrual health support, period guidance,
                and gentle reassurance.
              </p>
              <div className="ctaStack">
                <button className="primaryBtn wide" onClick={() => setShowLanding(false)}>
                  Open Chat Support
                </button>
                <button className="secondaryBtn wide" onClick={() => scrollToSection("features")}>
                  View Features
                </button>
              </div>
              <p className="finePrint">
                EmpowerHer provides general support and education. It does not replace medical care.
              </p>
            </div>

            <div className="benefitPanel">
              <h2>Why Join Us?</h2>
              <p>
                A private digital space built for menstrual health confidence, awareness, and emotional support.
              </p>
              <div className="benefitList">
                <div>
                  <strong>Private chat support</strong>
                  <span>Ask questions without fear of judgment or embarrassment.</span>
                </div>
                <div>
                  <strong>Educational resources</strong>
                  <span>Access guidance about periods, symptoms, hygiene, and healthy habits.</span>
                </div>
                <div>
                  <strong>Safe reassurance</strong>
                  <span>Receive calm responses designed for adolescents and young women.</span>
                </div>
                <div>
                  <strong>Always available</strong>
                  <span>Use the chatbot whenever you need support, day or night.</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <footer className="siteFooter">
          <div className="container footerGrid">
            <div>
              <h3>EmpowerHer</h3>
              <p>
                A safe, private, and emotionally supportive menstrual health chatbot designed to
                help adolescents and young women understand their bodies and receive reliable guidance.
              </p>
            </div>
            <div>
              <h3>Quick Links</h3>
              <button onClick={() => scrollToSection("home")}>Home</button>
              <button onClick={() => scrollToSection("about")}>About</button>
              <button onClick={() => scrollToSection("features")}>Features</button>
              <button onClick={() => setShowLanding(false)}>Chatbot</button>
              <button onClick={() => scrollToSection("support")}>Contact</button>
            </div>
            <div>
              <h3>Follow Us</h3>
              <span>Facebook</span>
              <span>Instagram</span>
              <span>Twitter (X)</span>
              <span>LinkedIn</span>
              <span>YouTube</span>
            </div>
          </div>
          <div className="copyrightBar">
            <span>© 2026 EmpowerHer. A final year project dedicated to adolescent menstrual health support.</span>
          </div>
        </footer>
      </div>
    );
  }

  return (
    <div className={`app ${chatView === "chat" ? "chatPage" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <div className="logo">
            <img src={appLogo} alt="EmpowerHer logo" />
          </div>
          <div>
            <h1>EmpowerHer</h1>
            <p>Private chat support for periods, moods, and menstrual health questions.</p>
          </div>
        </div>

        <div className="topActions">
          {chatView === "medical" && (
            <button className="ghost" onClick={() => setChatView("chat")}>
              Back to Chat
            </button>
          )}
          <button className="ghost" onClick={() => setShowLanding(true)}>
            Back to Home
          </button>
          <button
            className="ghost"
            onClick={() => {
              setMessages([initialBotMessage]);
              setChatView("chat");
            }}
          >
            Clear Chat
          </button>
        </div>
      </header>

      {chatView === "medical" ? (
        <main className="medicalShell">
          <section className="medicalHero card">
            <h2>Medical Information</h2>
            <p>
              General menstrual health guidance about doctors, clinics, hospitals, and when you should
              seek urgent help.
            </p>
          </section>

          <section className="medicalGrid">
            <article className="card medicalCard">
              <h3>When to See a Doctor or Clinic</h3>
              <ul className="medicalList">
                <li>Periods are missing for several months or are repeatedly very irregular.</li>
                <li>Cramps are severe or stop you from school, sleep, or normal activities.</li>
                <li>You have unusual discharge, strong smell, itching, burning, or pain.</li>
                <li>You feel worried about changes in bleeding, smell, or cycle timing.</li>
              </ul>
            </article>

            <article className="card medicalCard">
              <h3>When to Go to a Hospital Urgently</h3>
              <ul className="medicalList">
                <li>Very heavy bleeding that soaks pads quickly.</li>
                <li>Fainting, severe dizziness, chest pain, or extreme weakness.</li>
                <li>Strong pain with fever, vomiting, or feeling very unwell.</li>
                <li>Any situation where you feel unsafe or cannot manage at home.</li>
              </ul>
            </article>

            <article className="card medicalCard">
              <h3>What to Tell the Doctor</h3>
              <ul className="medicalList">
                <li>How long symptoms have been happening.</li>
                <li>How heavy the bleeding is and whether there are clots.</li>
                <li>If there is pain, fever, dizziness, smell, itching, or unusual discharge.</li>
                <li>Your age, the date of your last period, and what makes symptoms worse or better.</li>
              </ul>
            </article>

            <article className="card medicalCard">
              <h3>Finding Help</h3>
              <p className="muted">
                If symptoms are worrying, speak to a trusted adult, school nurse, clinic, family doctor,
                gynecology service, or the nearest hospital. For urgent symptoms, do not wait for the chatbot.
              </p>
              <div className="medicalActions">
                <button className="primaryBtn" onClick={() => setChatView("chat")}>
                  Back to Chat Support
                </button>
              </div>
            </article>

            <article className="card medicalCard medicalDoctors">
              <h3>Menstrual Health / Gynecology Doctors (Sri Lanka)</h3>
              <div className="doctorList">
                <div>
                  <strong>1. Dr. Nadira Dassanayake</strong>
                  <p>Obstetrics and Gynecology</p>
                  <p>Focus: Menstrual disorders, PCOS, fertility</p>
                  <p>Hospitals: Asiri Hospital, Nawaloka Hospital</p>
                  <p>Why important: Well-known consultant for women’s reproductive health</p>
                </div>
                <div>
                  <strong>2. Dr. Harsha Atapattu</strong>
                  <p>Consultant Obstetrician and Gynecologist</p>
                  <p>Focus: Hormonal issues, menstrual irregularities, adolescent gynecology</p>
                  <p>Hospitals: Lanka Hospitals</p>
                  <p>Strength: Strong experience with teenage and young women’s health</p>
                </div>
                <div>
                  <strong>3. Dr. Rishya Manikavasagar</strong>
                  <p>Obstetrics and Gynecology</p>
                  <p>Focus: Menstrual pain, reproductive health, counseling</p>
                  <p>Hospitals: Durdans Hospital</p>
                  <p>Why useful: Good for emotional and physical health combined care</p>
                </div>
                <div>
                  <strong>4. Dr. Kapila Jayaratne</strong>
                  <p>Obstetrics and Gynecology</p>
                  <p>Focus: PCOS, menstrual cycle management, fertility</p>
                  <p>Hospitals: Ninewells Hospital</p>
                  <p>Highlight: Popular among young women</p>
                </div>
                <div>
                  <strong>5. Dr. Shiromi Maduwage</strong>
                  <p>Women’s health and pregnancy care</p>
                  <p>Focus: Menstrual health education, reproductive wellbeing</p>
                  <p>Hospitals: Castle Street Hospital for Women</p>
                  <p>Why important: Government sector expertise</p>
                </div>
              </div>
            </article>

            <article className="card medicalCard">
              <h3>Government Clinics and Public Health Support</h3>
              <ul className="medicalList">
                <li><strong>Family Planning Association of Sri Lanka</strong> offers menstrual health advice, clinics, and education support.</li>
                <li><strong>Ministry of Health Sri Lanka</strong> provides public health services, education programs, and government clinic access.</li>
                <li>These services can be important for free or lower-cost menstrual health guidance and reproductive wellbeing support.</li>
              </ul>
            </article>
          </section>
        </main>
      ) : (
      <main className="main">
        <section className="chatArea">
          <div className="chatIntroCard">
            <div className="chatIntroTop">
              <div>
                <h2>Get Support</h2>
                <p>
                  Ask a question, describe a symptom, or continue a conversation in a private space.
                </p>
              </div>
              <div className="introBadge">Private and supportive</div>
            </div>
          </div>

          <div className="chatBox">
            {messages.map((m, i) => (
              <Bubble
                key={`${m.role}-${i}-${m.text.slice(0, 8)}`}
                role={m.role}
                text={m.text}
                meta={m.meta}
              />
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
              placeholder="Type your question here..."
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
            <p className="muted">Tap one to start faster.</p>
            <div className="suggestions">
              {suggestions.map((s) => (
                <button
                  key={s}
                  className="suggestionBtn"
                  onClick={() => void sendMessage(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>Safety Note</h2>
            <p className="muted">
              EmpowerHer offers general support, not medical diagnosis. If symptoms are severe,
              such as fainting, fever, or very heavy bleeding, please speak to a trusted adult or
              healthcare provider.
            </p>
          </div>

          <div className="card compactInfo medicalShortcut">
            <h2>Need Medical Info?</h2>
            <p className="muted">
              Read quick guidance about menstrual health, clinics, doctors, and hospital warning signs.
            </p>
            <button className="primaryBtn wide" onClick={() => setChatView("medical")}>
              Medical Info
            </button>
          </div>
        </aside>
      </main>
      )}
    </div>
  );
}
