import type { KeyboardEventHandler } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { User } from "firebase/auth";
import { AuthPanel } from "./components/AuthPanel";
import { isFirebaseConfigured } from "./lib/firebase";
import {
  signInWithGoogle,
  signInWithEmail,
  signOutUser,
  signUpWithEmail,
  subscribeToAuthChanges,
} from "./services/auth";
import {
  appendMessageToConversation,
  clearConversationMessages,
  createConversation,
  deleteConversation,
  listConversations,
  loadConversationMessages,
} from "./services/chatHistory";
import type {
  ChatMessage,
  ChatMeta,
  ChatResponse,
  ConversationSummary,
} from "./types/chat";
import "./App.css";
import appLogo from "../images/Logo.png";

type SectionKey = "home" | "about" | "features" | "why" | "support";
type AuthMode = "signin" | "signup";
type SymptomType =
  | "pain_cramps"
  | "late_period"
  | "heavy_bleeding"
  | "normal_discharge"
  | "mood_swings";

type SymptomCheckerState = {
  symptomType: SymptomType;
  duration: string;
  severity: "mild" | "moderate" | "severe";
  affectsSchoolOrSleep: boolean;
  fever: boolean;
  faintingOrDizziness: boolean;
  veryHeavyBleeding: boolean;
  itchingOrBurning: boolean;
  strongSmell: boolean;
  unusualColor: boolean;
  overthinkingOrPanic: boolean;
  notes: string;
};

const symptomTypeLabels: Record<SymptomType, string> = {
  pain_cramps: "Cramps or period pain",
  late_period: "Late or missed period",
  heavy_bleeding: "Heavy bleeding",
  normal_discharge: "Discharge or smell changes",
  mood_swings: "Mood changes",
};

const initialSymptomCheckerState: SymptomCheckerState = {
  symptomType: "pain_cramps",
  duration: "",
  severity: "moderate",
  affectsSchoolOrSleep: false,
  fever: false,
  faintingOrDizziness: false,
  veryHeavyBleeding: false,
  itchingOrBurning: false,
  strongSmell: false,
  unusualColor: false,
  overthinkingOrPanic: false,
  notes: "",
};

const initialBotMessage: ChatMessage = {
  role: "bot",
  text: "Hi, I'm EmpowerHer. I'm here to listen and help with questions about periods, moods, or anything you're feeling. What's on your mind today?",
};

function Bubble({ role, text, meta }: ChatMessage) {
  const isUrgent = meta?.escalation_level === "urgent" || meta?.escalation_level === "critical";

  return (
    <div className={`row ${role}`}>
      <div className={`bubble ${role}`}>
        <div className="bubbleText">{text}</div>
        {role === "bot" && isUrgent && (
          <div className="escalationBanner">
            <strong>
              {meta?.escalation_level === "critical" ? "Urgent safety action needed" : "Red-flag symptoms detected"}
            </strong>
            <span>
              Tell a trusted adult and seek in-person care instead of relying only on chat.
            </span>
          </div>
        )}
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
            <span className="chip">
              Escalation: {meta.escalation_level ?? "-"}
            </span>
            <span className="chip">
              Flags:{" "}
              {meta.escalation_reasons && meta.escalation_reasons.length
                ? meta.escalation_reasons.join(", ")
                : "-"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function hasUserConversation(history: ChatMessage[]) {
  return history.some((message) => message.role === "user");
}

function createConversationId() {
  return `conversation-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function buildSymptomCheckerPrompt(form: SymptomCheckerState) {
  const details: string[] = [];
  details.push(`Main issue: ${symptomTypeLabels[form.symptomType]}.`);
  details.push(`Severity: ${form.severity}.`);

  if (form.duration.trim()) {
    details.push(`Duration: ${form.duration.trim()}.`);
  }
  if (form.affectsSchoolOrSleep) {
    details.push("It affects school, sleep, or normal activities.");
  }
  if (form.fever) {
    details.push("I also have fever or feel very unwell.");
  }
  if (form.faintingOrDizziness) {
    details.push("I feel dizzy, faint, or nearly faint.");
  }
  if (form.veryHeavyBleeding) {
    details.push("Bleeding is very heavy or I am soaking pads quickly.");
  }
  if (form.itchingOrBurning) {
    details.push("There is itching or burning.");
  }
  if (form.strongSmell) {
    details.push("There is a strong or unusual smell.");
  }
  if (form.unusualColor) {
    details.push("The discharge color is unusual, such as yellow or green.");
  }
  if (form.overthinkingOrPanic) {
    details.push("I feel panicky, overwhelmed, or I cannot calm down.");
  }
  if (form.notes.trim()) {
    details.push(`Extra details: ${form.notes.trim()}.`);
  }

  return `Symptom checker summary: ${details.join(" ")} Please tell me what this could mean, what I can do safely at home, and whether I should tell a trusted adult or go to a clinic.`;
}

function buildSummary(conversationId: string, messages: ChatMessage[]): ConversationSummary {
  const firstUser = messages.find((message) => message.role === "user");
  const lastMessage = messages[messages.length - 1];
  const titleSource = firstUser?.text || "New conversation";
  const previewSource = lastMessage?.text || "";

  return {
    id: conversationId,
    title:
      titleSource.length > 48 ? `${titleSource.slice(0, 48).trim()}...` : titleSource,
    preview:
      previewSource.length > 80
        ? `${previewSource.slice(0, 80).trim()}...`
        : previewSource,
    messageCount: messages.length,
  };
}

export default function App() {
  const [booting, setBooting] = useState(true);
  const [showLanding, setShowLanding] = useState(true);
  const [chatView, setChatView] = useState<"chat" | "medical">("chat");
  const [messages, setMessages] = useState<ChatMessage[]>([initialBotMessage]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [authBusy, setAuthBusy] = useState(false);
  const [authReady, setAuthReady] = useState(!isFirebaseConfigured);
  const [historyReady, setHistoryReady] = useState(!isFirebaseConfigured);
  const [authMessage, setAuthMessage] = useState<string | null>(null);
  const [symptomChecker, setSymptomChecker] = useState<SymptomCheckerState>(
    initialSymptomCheckerState
  );
  const [conversationSummaries, setConversationSummaries] = useState<
    ConversationSummary[]
  >([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    null
  );

  const endRef = useRef<HTMLDivElement | null>(null);
  const messagesRef = useRef<ChatMessage[]>([initialBotMessage]);
  const saveQueueRef = useRef<Map<string, Promise<void>>>(new Map());
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
  const latestBotMeta = [...messages]
    .reverse()
    .find((message) => message.role === "bot" && message.meta)?.meta;

  useEffect(() => {
    messagesRef.current = messages;
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const timer = window.setTimeout(() => setBooting(false), 1200);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!isFirebaseConfigured) {
      return undefined;
    }

    const unsubscribe = subscribeToAuthChanges(async (nextUser) => {
      setUser(nextUser);
      setAuthReady(true);
      setHistoryReady(false);
      setAuthMessage(null);

      if (!nextUser) {
        setConversationSummaries([]);
        setActiveConversationId(null);
        setMessages([initialBotMessage]);
        setHistoryReady(true);
        return;
      }

      try {
        const summaries = await listConversations(nextUser.uid);

        if (summaries.length > 0) {
          setConversationSummaries(summaries);
          setActiveConversationId(summaries[0].id);
          const remoteMessages = await loadConversationMessages(
            nextUser.uid,
            summaries[0].id
          );
          setMessages(remoteMessages.length > 0 ? remoteMessages : [initialBotMessage]);
        } else if (hasUserConversation(messagesRef.current)) {
          const conversationId = createConversationId();
          await createConversation(nextUser.uid, conversationId, messagesRef.current);
          setConversationSummaries([buildSummary(conversationId, messagesRef.current)]);
          setActiveConversationId(conversationId);
        } else {
          setConversationSummaries([]);
          setActiveConversationId(null);
          setMessages([initialBotMessage]);
        }
      } catch (error) {
        console.error(error);
        setAuthMessage(
          "Authentication worked, but loading saved messages failed. Check Firestore setup and security rules."
        );
      } finally {
        setHistoryReady(true);
      }
    });

    return unsubscribe;
  }, []);

  const scrollToSection = (key: SectionKey) => {
    sectionRefs[key].current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  const upsertSummary = (summary: ConversationSummary) => {
    setConversationSummaries((prev) => [
      summary,
      ...prev.filter((item) => item.id !== summary.id),
    ]);
  };

  const persistConversationMessage = (
    conversationId: string,
    message: ChatMessage,
    nextMessages: ChatMessage[]
  ) => {
    if (!user) {
      return Promise.resolve();
    }

    const summary = buildSummary(conversationId, nextMessages);
    upsertSummary(summary);

    const previousTask = saveQueueRef.current.get(conversationId) ?? Promise.resolve();
    const nextTask = previousTask
      .catch(() => undefined)
      .then(async () => {
        await appendMessageToConversation(
          user.uid,
          conversationId,
          message,
          nextMessages
        );
      })
      .catch((error) => {
        console.error(error);
        setAuthMessage(
          "Signed in, but Firestore could not save this conversation. Check database rules and indexes."
        );
      });

    saveQueueRef.current.set(conversationId, nextTask);
    return nextTask;
  };

  const ensureActiveConversation = async (seedMessages: ChatMessage[]) => {
    if (!user) {
      return null;
    }

    if (activeConversationId) {
      return activeConversationId;
    }

    const conversationId = createConversationId();

    try {
      await createConversation(user.uid, conversationId, seedMessages);
      setActiveConversationId(conversationId);
      upsertSummary(buildSummary(conversationId, seedMessages));
      return conversationId;
    } catch (error) {
      console.error(error);
      setAuthMessage(
        "Could not create a new saved conversation. Check Firestore setup."
      );
      return null;
    }
  };

  const startNewConversation = async () => {
    setChatView("chat");
    setAuthMessage(null);

    if (!user) {
      setMessages([initialBotMessage]);
      return;
    }

    const conversationId = createConversationId();

    try {
      await createConversation(user.uid, conversationId, [initialBotMessage]);
      setActiveConversationId(conversationId);
      setMessages([initialBotMessage]);
      upsertSummary(buildSummary(conversationId, [initialBotMessage]));
      saveQueueRef.current.set(conversationId, Promise.resolve());
    } catch (error) {
      console.error(error);
      setAuthMessage("Could not create a new saved conversation.");
    }
  };

  const switchConversation = async (conversationId: string) => {
    if (!user || conversationId === activeConversationId) {
      return;
    }

    setHistoryReady(false);
    setAuthMessage(null);
    setMessages([initialBotMessage]);

    try {
      const nextMessages = await loadConversationMessages(user.uid, conversationId);
      setActiveConversationId(conversationId);
      setMessages(nextMessages.length > 0 ? nextMessages : [initialBotMessage]);
      setChatView("chat");
    } catch (error) {
      console.error(error);
      setAuthMessage("Could not open that saved conversation.");
    } finally {
      setHistoryReady(true);
    }
  };

  const clearChat = async () => {
    setMessages([initialBotMessage]);
    setChatView("chat");

    if (!user || !activeConversationId) {
      return;
    }

    try {
      await clearConversationMessages(user.uid, activeConversationId, [
        initialBotMessage,
      ]);
      upsertSummary(buildSummary(activeConversationId, [initialBotMessage]));
      setAuthMessage("Saved conversation cleared.");
    } catch (error) {
      console.error(error);
      setAuthMessage("Could not clear saved chat history from Firestore.");
    }
  };

  const removeConversation = async () => {
    if (!user || !activeConversationId) {
      return;
    }

    const conversationId = activeConversationId;
    const previousMessages = messagesRef.current;
    const previousSummaries = conversationSummaries;
    const remainingSummaries = conversationSummaries.filter(
      (summary) => summary.id !== conversationId
    );
    const nextConversationId =
      remainingSummaries.length > 0 ? remainingSummaries[0].id : null;

    setHistoryReady(false);
    setMessages([initialBotMessage]);
    setConversationSummaries(remainingSummaries);
    setActiveConversationId(nextConversationId);

    try {
      await deleteConversation(user.uid, conversationId);
      saveQueueRef.current.delete(conversationId);

      if (nextConversationId) {
        const nextMessages = await loadConversationMessages(
          user.uid,
          nextConversationId
        );
        setMessages(nextMessages.length > 0 ? nextMessages : [initialBotMessage]);
      }

      setAuthMessage("Conversation deleted.");
    } catch (error) {
      console.error(error);
      setAuthMessage("Could not delete the selected conversation.");
      setConversationSummaries(previousSummaries);
      setActiveConversationId(conversationId);
      setMessages(previousMessages);
    } finally {
      setHistoryReady(true);
    }
  };

  const updateSymptomChecker = <K extends keyof SymptomCheckerState>(
    key: K,
    value: SymptomCheckerState[K]
  ) => {
    setSymptomChecker((prev) => ({ ...prev, [key]: value }));
  };

  const submitSymptomChecker = async () => {
    await sendMessage(buildSymptomCheckerPrompt(symptomChecker));
  };

  const sendMessage = async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg || loading) return;

    const currentMessages = messagesRef.current;
    const userMessage: ChatMessage = { role: "user", text: msg };
    const optimisticMessages = [...currentMessages, userMessage];
    const historyForRequest = currentMessages.slice(-6);

    setMessages(optimisticMessages);
    setInput("");
    setLoading(true);

    const conversationId = await ensureActiveConversation(currentMessages);
    if (conversationId) {
      void persistConversationMessage(
        conversationId,
        userMessage,
        optimisticMessages
      );
    }

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          history: historyForRequest,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data: ChatResponse = await res.json();
      const reply =
        data.reply?.trim() || "Sorry, I could not generate a reply right now.";

      const meta: ChatMeta = {
        intent: data.intent ?? undefined,
        topic: data.topic ?? undefined,
        emotions: Array.isArray(data.emotions) ? data.emotions : undefined,
        kb_sources: Array.isArray(data.kb_sources)
          ? data.kb_sources
          : undefined,
        escalation_level: data.escalation_level ?? undefined,
        escalation_reasons: Array.isArray(data.escalation_reasons)
          ? data.escalation_reasons
          : undefined,
      };

      const botMessage: ChatMessage = { role: "bot", text: reply, meta };
      const finalMessages = [...optimisticMessages, botMessage];
      setMessages(finalMessages);

      if (conversationId) {
        void persistConversationMessage(
          conversationId,
          botMessage,
          finalMessages
        );
      }
    } catch (error) {
      console.error(error);
      const fallbackBotMessage: ChatMessage = {
        role: "bot",
        text: "Oops, I could not connect to the backend. Is Flask running on http://127.0.0.1:5000 ?",
      };
      const fallbackMessages = [...optimisticMessages, fallbackBotMessage];
      setMessages(fallbackMessages);

      if (conversationId) {
        void persistConversationMessage(
          conversationId,
          fallbackBotMessage,
          fallbackMessages
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleEmailAuth = async (
    mode: AuthMode,
    email: string,
    password: string
  ) => {
    setAuthBusy(true);
    setAuthMessage(null);

    try {
      if (mode === "signup") {
        await signUpWithEmail(email, password);
        setAuthMessage("Account created. Your chat history will now be saved.");
      } else {
        await signInWithEmail(email, password);
        setAuthMessage("Signed in. Saved chat history is available on this device.");
      }
    } catch (error) {
      console.error(error);
      setAuthMessage(
        error instanceof Error ? error.message : "Authentication failed."
      );
    } finally {
      setAuthBusy(false);
    }
  };

  const handleGoogleAuth = async () => {
    setAuthBusy(true);
    setAuthMessage(null);

    try {
      await signInWithGoogle();
      setAuthMessage("Signed in with Google.");
    } catch (error) {
      console.error(error);
      setAuthMessage(
        error instanceof Error ? error.message : "Google sign-in failed."
      );
    } finally {
      setAuthBusy(false);
    }
  };

  const handleSignOut = async () => {
    setAuthBusy(true);
    setAuthMessage(null);

    try {
      await signOutUser();
      setAuthMessage("Signed out. Guest chats will stay local only.");
    } catch (error) {
      console.error(error);
      setAuthMessage(error instanceof Error ? error.message : "Could not sign out.");
    } finally {
      setAuthBusy(false);
    }
  };

  const onKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  const statusText = !isFirebaseConfigured
    ? "Firebase is not configured yet. Chat works in guest mode only."
    : !authReady || !historyReady
      ? "Checking account and saved conversation..."
      : user
        ? `Signed in as ${user.email ?? "Google user"}`
        : "Guest mode. Sign in to save chat history.";

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
                <div className="featureIcon">💞</div>
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
            <span>Copyright 2026 EmpowerHer. A final year project dedicated to adolescent menstrual health support.</span>
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
          <button className="ghost" onClick={() => void clearChat()}>
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
                  <p>Why important: Well-known consultant for women's reproductive health</p>
                </div>
                <div>
                  <strong>2. Dr. Harsha Atapattu</strong>
                  <p>Consultant Obstetrician and Gynecologist</p>
                  <p>Focus: Hormonal issues, menstrual irregularities, adolescent gynecology</p>
                  <p>Hospitals: Lanka Hospitals</p>
                  <p>Strength: Strong experience with teenage and young women's health</p>
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
                  <p>Women's health and pregnancy care</p>
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
                <div className="introBadge">
                  {user ? "History syncing" : "Private and supportive"}
                </div>
              </div>
              <div className="statusBar">{statusText}</div>
              {authMessage && <div className="statusNote">{authMessage}</div>}
              {latestBotMeta?.escalation_level &&
                latestBotMeta.escalation_level !== "none" && (
                  <div className="escalationCard">
                    <strong>
                      {latestBotMeta.escalation_level === "critical"
                        ? "Immediate safety support needed"
                        : "Urgent medical follow-up recommended"}
                    </strong>
                    <span>
                      {latestBotMeta.escalation_reasons?.length
                        ? `Detected: ${latestBotMeta.escalation_reasons.join(", ")}.`
                        : "This conversation includes red-flag symptoms."}{" "}
                      Please tell a trusted adult and seek clinic or hospital care if symptoms are severe.
                    </span>
                  </div>
                )}
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
                disabled={!historyReady}
              />
              <button
                className="sendBtn"
                disabled={loading || !input.trim() || !historyReady}
                onClick={() => void sendMessage()}
              >
                Send
              </button>
            </div>
          </section>

          <aside className="sidePanel">
            <AuthPanel
              authBusy={authBusy}
              authEnabled={isFirebaseConfigured}
              authReady={authReady}
              historyReady={historyReady}
              user={user}
              onEmailAuth={handleEmailAuth}
              onGoogleAuth={handleGoogleAuth}
              onSignOut={handleSignOut}
            />

            {user && (
              <div className="card conversationCard">
                <div className="conversationHeader">
                  <h2>Saved Chats</h2>
                  <button
                    className="ghost conversationAction"
                    onClick={() => void startNewConversation()}
                    disabled={!historyReady || loading}
                  >
                    New
                  </button>
                </div>
                <div className="conversationList">
                  {conversationSummaries.length === 0 ? (
                    <p className="muted">
                      No saved chats yet. Start a new conversation and it will appear here.
                    </p>
                  ) : (
                    conversationSummaries.map((summary) => (
                      <button
                        key={summary.id}
                        className={
                          summary.id === activeConversationId
                            ? "conversationItem active"
                            : "conversationItem"
                        }
                        onClick={() => void switchConversation(summary.id)}
                        disabled={!historyReady || loading}
                      >
                        <strong>{summary.title}</strong>
                        <span>{summary.preview || "No preview yet."}</span>
                        <small>{summary.messageCount} messages</small>
                      </button>
                    ))
                  )}
                </div>
                {activeConversationId && (
                  <button
                    className="ghost wide deleteConversationBtn"
                    onClick={() => void removeConversation()}
                    disabled={!historyReady || loading}
                  >
                    Delete Selected Chat
                  </button>
                )}
              </div>
            )}

            <div className="card">
              <h2>Symptom Checker</h2>
              <p className="muted">
                Use the guided form if you want a more structured answer than free-text chat.
              </p>
              <div className="symptomForm">
                <label className="symptomField">
                  <span>Main symptom</span>
                  <select
                    value={symptomChecker.symptomType}
                    onChange={(e) =>
                      updateSymptomChecker("symptomType", e.target.value as SymptomType)
                    }
                    disabled={!historyReady || loading}
                  >
                    {Object.entries(symptomTypeLabels).map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="symptomSplit">
                  <label className="symptomField">
                    <span>Severity</span>
                    <select
                      value={symptomChecker.severity}
                      onChange={(e) =>
                        updateSymptomChecker(
                          "severity",
                          e.target.value as SymptomCheckerState["severity"]
                        )
                      }
                      disabled={!historyReady || loading}
                    >
                      <option value="mild">Mild</option>
                      <option value="moderate">Moderate</option>
                      <option value="severe">Severe</option>
                    </select>
                  </label>

                  <label className="symptomField">
                    <span>Duration</span>
                    <input
                      type="text"
                      value={symptomChecker.duration}
                      onChange={(e) => updateSymptomChecker("duration", e.target.value)}
                      placeholder="e.g. 2 days, 1 week"
                      disabled={!historyReady || loading}
                    />
                  </label>
                </div>

                <div className="symptomChecks">
                  <label><input type="checkbox" checked={symptomChecker.affectsSchoolOrSleep} onChange={(e) => updateSymptomChecker("affectsSchoolOrSleep", e.target.checked)} disabled={!historyReady || loading} />Affects school or sleep</label>
                  <label><input type="checkbox" checked={symptomChecker.fever} onChange={(e) => updateSymptomChecker("fever", e.target.checked)} disabled={!historyReady || loading} />Fever or very unwell</label>
                  <label><input type="checkbox" checked={symptomChecker.faintingOrDizziness} onChange={(e) => updateSymptomChecker("faintingOrDizziness", e.target.checked)} disabled={!historyReady || loading} />Fainting or dizziness</label>
                  <label><input type="checkbox" checked={symptomChecker.veryHeavyBleeding} onChange={(e) => updateSymptomChecker("veryHeavyBleeding", e.target.checked)} disabled={!historyReady || loading} />Very heavy bleeding</label>
                  <label><input type="checkbox" checked={symptomChecker.itchingOrBurning} onChange={(e) => updateSymptomChecker("itchingOrBurning", e.target.checked)} disabled={!historyReady || loading} />Itching or burning</label>
                  <label><input type="checkbox" checked={symptomChecker.strongSmell} onChange={(e) => updateSymptomChecker("strongSmell", e.target.checked)} disabled={!historyReady || loading} />Strong smell</label>
                  <label><input type="checkbox" checked={symptomChecker.unusualColor} onChange={(e) => updateSymptomChecker("unusualColor", e.target.checked)} disabled={!historyReady || loading} />Yellow or green discharge</label>
                  <label><input type="checkbox" checked={symptomChecker.overthinkingOrPanic} onChange={(e) => updateSymptomChecker("overthinkingOrPanic", e.target.checked)} disabled={!historyReady || loading} />Panicky or overwhelmed</label>
                </div>

                <label className="symptomField">
                  <span>Extra notes</span>
                  <textarea
                    value={symptomChecker.notes}
                    onChange={(e) => updateSymptomChecker("notes", e.target.value)}
                    placeholder="Anything else important?"
                    rows={3}
                    disabled={!historyReady || loading}
                  />
                </label>

                <button
                  className="primaryBtn wide"
                  onClick={() => void submitSymptomChecker()}
                  disabled={!historyReady || loading}
                >
                  Check Symptoms
                </button>
              </div>
            </div>

            <div className="card">
              <h2>Quick Prompts</h2>
              <p className="muted">Tap one to start faster.</p>
              <div className="suggestions">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    className="suggestionBtn"
                    onClick={() => void sendMessage(s)}
                    disabled={!historyReady || loading}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            <div className="card">
              <h2>Red-Flag Escalation</h2>
              <p className="muted">
                If there is severe pain, fainting, fever, soaking pads quickly, chest pain, or you
                feel unsafe, do not rely only on chat. Tell a trusted adult and go to a clinic or
                hospital.
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
