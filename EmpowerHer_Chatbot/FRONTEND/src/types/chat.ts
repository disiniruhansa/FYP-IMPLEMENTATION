export type ChatMeta = {
  intent?: string;
  topic?: string;
  emotions?: string[];
  kb_sources?: string[];
};

export type ChatMessage = {
  role: "user" | "bot";
  text: string;
  meta?: ChatMeta;
};

export type ChatResponse = {
  reply?: string;
  emotions?: string[];
  raw_emotions?: unknown;
  topic?: string;
  intent?: string;
  kb_sources?: string[];
};

export type ConversationSummary = {
  id: string;
  title: string;
  preview: string;
  messageCount: number;
};
