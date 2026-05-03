import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  getDocs,
  limit,
  orderBy,
  query,
  serverTimestamp,
  setDoc,
} from "firebase/firestore";
import { db } from "./firebase.ts";
import type { ChatMessage, ConversationSummary } from "./chat.ts";

type StoredMessage = {
  role?: unknown;
  text?: unknown;
  meta?: unknown;
};

type StoredConversation = {
  title?: unknown;
  preview?: unknown;
  messageCount?: unknown;
};

type FirestoreMessage = {
  role: ChatMessage["role"];
  text: string;
  meta?: Record<string, string | string[]>;
};

type LocalConversationRecord = {
  summary: ConversationSummary;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
};

type LocalConversationMap = Record<string, LocalConversationRecord>;

function requireDb() {
  if (!db) {
    throw new Error("Firestore is not configured.");
  }

  return db;
}

function conversationsCollection(uid: string) {
  return collection(requireDb(), "users", uid, "conversations");
}

function conversationDoc(uid: string, conversationId: string) {
  return doc(requireDb(), "users", uid, "conversations", conversationId);
}

function messagesCollection(uid: string, conversationId: string) {
  return collection(
    requireDb(),
    "users",
    uid,
    "conversations",
    conversationId,
    "messages"
  );
}

function localStorageKey(uid: string) {
  return `empowerher:conversations:${uid}`;
}

function canUseLocalStorage() {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function readLocalConversations(uid: string): LocalConversationMap {
  if (!canUseLocalStorage()) {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(localStorageKey(uid));
    if (!raw) {
      return {};
    }

    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === "object"
      ? (parsed as LocalConversationMap)
      : {};
  } catch {
    return {};
  }
}

function writeLocalConversations(uid: string, conversations: LocalConversationMap) {
  if (!canUseLocalStorage()) {
    return;
  }

  window.localStorage.setItem(localStorageKey(uid), JSON.stringify(conversations));
}

function sanitizeMessage(record: StoredMessage): ChatMessage | null {
  if (
    (record.role !== "user" && record.role !== "bot") ||
    typeof record.text !== "string"
  ) {
    return null;
  }

  const metaCandidate = record.meta;
  const meta =
    metaCandidate && typeof metaCandidate === "object"
      ? (metaCandidate as ChatMessage["meta"])
      : undefined;

  return {
    role: record.role,
    text: record.text,
    meta,
  };
}

function serializeMessage(message: ChatMessage): FirestoreMessage {
  const metaEntries = Object.entries(message.meta ?? {}).filter(([, value]) => {
    if (Array.isArray(value)) {
      return value.length > 0;
    }

    return value !== undefined;
  });

  const meta =
    metaEntries.length > 0
      ? Object.fromEntries(metaEntries) as Record<string, string | string[]>
      : undefined;

  return {
    role: message.role,
    text: message.text,
    ...(meta ? { meta } : {}),
  };
}

function sanitizeSummary(id: string, record: StoredConversation): ConversationSummary {
  const title =
    typeof record.title === "string" && record.title.trim()
      ? record.title.trim()
      : "Untitled conversation";
  const preview =
    typeof record.preview === "string" ? record.preview.trim() : "";
  const messageCount =
    typeof record.messageCount === "number" ? record.messageCount : 0;

  return {
    id,
    title,
    preview,
    messageCount,
  };
}

function buildConversationSummary(messages: ChatMessage[]) {
  const firstUserMessage = messages.find((message) => message.role === "user");
  const lastMessage = messages[messages.length - 1];
  const titleSource = firstUserMessage?.text || "New conversation";
  const previewSource = lastMessage?.text || "";

  return {
    title:
      titleSource.length > 48 ? `${titleSource.slice(0, 48).trim()}...` : titleSource,
    preview:
      previewSource.length > 80
        ? `${previewSource.slice(0, 80).trim()}...`
        : previewSource,
    messageCount: messages.length,
  };
}

function upsertLocalConversation(
  uid: string,
  conversationId: string,
  messages: ChatMessage[]
) {
  const conversations = readLocalConversations(uid);
  const existing = conversations[conversationId];
  const now = Date.now();

  conversations[conversationId] = {
    summary: {
      id: conversationId,
      ...buildConversationSummary(messages),
    },
    messages,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  };

  writeLocalConversations(uid, conversations);
}

function listLocalConversations(uid: string) {
  const conversations = readLocalConversations(uid);

  return Object.entries(conversations)
    .sort(([, left], [, right]) => right.updatedAt - left.updatedAt)
    .map(([id, record]) => ({
      id,
      title: record.summary.title,
      preview: record.summary.preview,
      messageCount: record.messages.length,
    }));
}

function loadLocalConversationMessages(uid: string, conversationId: string) {
  const conversations = readLocalConversations(uid);
  return conversations[conversationId]?.messages ?? [];
}

function deleteLocalConversation(uid: string, conversationId: string) {
  const conversations = readLocalConversations(uid);
  delete conversations[conversationId];
  writeLocalConversations(uid, conversations);
}

async function writeConversationMetadata(
  uid: string,
  conversationId: string,
  messages: ChatMessage[]
) {
  const summary = buildConversationSummary(messages);

  await setDoc(
    conversationDoc(uid, conversationId),
    {
      ...summary,
      updatedAt: serverTimestamp(),
      createdAt: serverTimestamp(),
    },
    { merge: true }
  );
}

async function deleteConversationMessages(uid: string, conversationId: string) {
  const snapshot = await getDocs(messagesCollection(uid, conversationId));
  await Promise.all(snapshot.docs.map((docSnapshot) => deleteDoc(docSnapshot.ref)));
}

export async function createConversation(
  uid: string,
  conversationId: string,
  messages: ChatMessage[]
) {
  upsertLocalConversation(uid, conversationId, messages);

  try {
    await replaceConversationMessages(uid, conversationId, messages);
  } catch {
    // Fall back to browser storage when Firestore is unavailable.
  }
}

export async function appendMessageToConversation(
  uid: string,
  conversationId: string,
  message: ChatMessage,
  nextMessages?: ChatMessage[]
) {
  const localNextMessages =
    nextMessages ??
    [...loadLocalConversationMessages(uid, conversationId), message];
  upsertLocalConversation(uid, conversationId, localNextMessages);

  try {
    await addDoc(messagesCollection(uid, conversationId), {
      ...serializeMessage(message),
      clientCreatedAt: Date.now(),
      createdAt: serverTimestamp(),
    });

    if (nextMessages) {
      await writeConversationMetadata(uid, conversationId, nextMessages);
      return;
    }

    const currentMessages = await loadConversationMessages(uid, conversationId);
    await writeConversationMetadata(uid, conversationId, currentMessages);
  } catch {
    // Local storage already contains the latest conversation state.
  }
}

export async function loadConversationMessages(
  uid: string,
  conversationId: string
) {
  try {
    const snapshot = await getDocs(
      query(messagesCollection(uid, conversationId), orderBy("clientCreatedAt", "asc"))
    );

    const messages = snapshot.docs
      .map((docSnapshot) => sanitizeMessage(docSnapshot.data() as StoredMessage))
      .filter((message): message is ChatMessage => message !== null);

    if (messages.length > 0) {
      upsertLocalConversation(uid, conversationId, messages);
      return messages;
    }
  } catch {
    // Fall back to browser storage below.
  }

  return loadLocalConversationMessages(uid, conversationId);
}

export async function listConversations(uid: string) {
  const localSummaries = listLocalConversations(uid);

  try {
    const snapshot = await getDocs(
      query(conversationsCollection(uid), orderBy("updatedAt", "desc"), limit(20))
    );

    const remoteSummaries = snapshot.docs.map((docSnapshot) =>
      sanitizeSummary(docSnapshot.id, docSnapshot.data() as StoredConversation)
    );

    return remoteSummaries.length > 0 ? remoteSummaries : localSummaries;
  } catch {
    return localSummaries;
  }
}

export async function clearConversationMessages(
  uid: string,
  conversationId: string,
  fallbackMessages: ChatMessage[]
) {
  upsertLocalConversation(uid, conversationId, fallbackMessages);

  try {
    await deleteConversationMessages(uid, conversationId);
    await replaceConversationMessages(uid, conversationId, fallbackMessages);
  } catch {
    // Local storage already has the cleared conversation.
  }
}

export async function replaceConversationMessages(
  uid: string,
  conversationId: string,
  messages: ChatMessage[]
) {
  upsertLocalConversation(uid, conversationId, messages);

  try {
    await deleteConversationMessages(uid, conversationId);

    await Promise.all(
      messages.map((message, index) =>
        addDoc(messagesCollection(uid, conversationId), {
          ...serializeMessage(message),
          clientCreatedAt: Date.now() + index,
          createdAt: serverTimestamp(),
        })
      )
    );

    await writeConversationMetadata(uid, conversationId, messages);
  } catch {
    // Local storage already contains the latest state.
  }
}

export async function deleteConversation(uid: string, conversationId: string) {
  deleteLocalConversation(uid, conversationId);

  try {
    await deleteConversationMessages(uid, conversationId);
    await deleteDoc(conversationDoc(uid, conversationId));
  } catch {
    // Local deletion is already complete.
  }
}
