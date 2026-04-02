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
import { db } from "../lib/firebase";
import type { ChatMessage, ConversationSummary } from "../types/chat";

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
  await replaceConversationMessages(uid, conversationId, messages);
}

export async function appendMessageToConversation(
  uid: string,
  conversationId: string,
  message: ChatMessage,
  nextMessages?: ChatMessage[]
) {
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
}

export async function loadConversationMessages(
  uid: string,
  conversationId: string
) {
  const snapshot = await getDocs(
    query(messagesCollection(uid, conversationId), orderBy("clientCreatedAt", "asc"))
  );

  return snapshot.docs
    .map((docSnapshot) => sanitizeMessage(docSnapshot.data() as StoredMessage))
    .filter((message): message is ChatMessage => message !== null);
}

export async function listConversations(uid: string) {
  const snapshot = await getDocs(
    query(conversationsCollection(uid), orderBy("updatedAt", "desc"), limit(20))
  );

  return snapshot.docs.map((docSnapshot) =>
    sanitizeSummary(docSnapshot.id, docSnapshot.data() as StoredConversation)
  );
}

export async function clearConversationMessages(
  uid: string,
  conversationId: string,
  fallbackMessages: ChatMessage[]
) {
  await deleteConversationMessages(uid, conversationId);
  await replaceConversationMessages(uid, conversationId, fallbackMessages);
}

export async function replaceConversationMessages(
  uid: string,
  conversationId: string,
  messages: ChatMessage[]
) {
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
}

export async function deleteConversation(uid: string, conversationId: string) {
  await deleteConversationMessages(uid, conversationId);
  await deleteDoc(conversationDoc(uid, conversationId));
}
