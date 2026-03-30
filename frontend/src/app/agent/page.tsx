"use client";
import { useState, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import { Dataset, ChatMessage } from "@/types";

const DATASET_KEY = "automl_agent_dataset";

function chatKey(datasetId: string | undefined): string {
  return `automl_agent_chat_${datasetId || "global"}`;
}

function sessionKey(datasetId: string | undefined): string {
  return `automl_agent_session_${datasetId || "global"}`;
}

function readStoredValue(key: string): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(key) ?? sessionStorage.getItem(key);
}

function writeStoredValue(key: string, value: string) {
  if (typeof window === "undefined") return;
  localStorage.setItem(key, value);
  sessionStorage.removeItem(key);
}

function removeStoredValue(key: string) {
  if (typeof window === "undefined") return;
  localStorage.removeItem(key);
  sessionStorage.removeItem(key);
}

function loadMessages(datasetId: string | undefined): ChatMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = readStoredValue(chatKey(datasetId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return parsed.map((m: any) => ({ ...m, timestamp: new Date(m.timestamp) }));
  } catch {
    return [];
  }
}

function saveMessages(messages: ChatMessage[], datasetId: string | undefined) {
  try {
    writeStoredValue(chatKey(datasetId), JSON.stringify(messages));
  } catch {}
}

function loadDatasetId(): string | undefined {
  if (typeof window === "undefined") return undefined;
  return readStoredValue(DATASET_KEY) || undefined;
}

function saveDatasetId(id: string | undefined) {
  try {
    if (id) writeStoredValue(DATASET_KEY, id);
    else removeStoredValue(DATASET_KEY);
  } catch {}
}

function getSessionId(datasetId: string | undefined): string {
  if (typeof window === "undefined") return sessionKey(datasetId);
  const key = sessionKey(datasetId);
  const existing = readStoredValue(key);
  if (existing) return existing;
  writeStoredValue(key, key);
  return key;
}

export default function AgentPage() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadMessages(loadDatasetId()));
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [activeDataset, setActiveDataset] = useState<string | undefined>(() => loadDatasetId());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.listDatasets().then((ds) => {
      setDatasets(ds);
      if (!activeDataset && ds.length > 0) {
        setActiveDataset(ds[0].id);
        saveDatasetId(ds[0].id);
      }
    }).catch(() => {});
  }, []);

  // Persist messages to sessionStorage (per dataset)
  useEffect(() => {
    saveMessages(messages, activeDataset);
  }, [messages, activeDataset]);

  // When dataset changes, load that dataset's chat history
  useEffect(() => {
    saveDatasetId(activeDataset);
    setMessages(loadMessages(activeDataset));
  }, [activeDataset]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || sending) return;

    const userMsg: ChatMessage = {
      role: "user",
      content: text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);

    try {
      const result = await api.chat(text, getSessionId(activeDataset), activeDataset);
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: result.response,
        tool_calls: result.tool_calls,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e: any) {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: `Sorry, I encountered an error: ${e.message}. Please check that the backend is running.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  async function handleReset() {
    await api.resetAgent(getSessionId(activeDataset)).catch(() => {});
    setMessages([]);
    removeStoredValue(chatKey(activeDataset));
    removeStoredValue(sessionKey(activeDataset));
  }

  const activeDsName = datasets.find((d) => d.id === activeDataset)?.name;

  return (
    <div className="flex flex-col min-h-screen p-6 max-w-7xl mx-auto">
      <div className="relative overflow-hidden rounded-[2rem] border border-slate-200/80 bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.16),_transparent_24%),radial-gradient(circle_at_bottom_right,_rgba(16,185,129,0.12),_transparent_18%),linear-gradient(135deg,_rgba(255,255,255,0.95),_rgba(247,250,252,0.9))] px-8 py-8 mb-6 shadow-[0_24px_70px_rgba(15,23,42,0.07)]">
        <div className="max-w-3xl">
          <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-600 mb-3">AI Workspace</p>
          <h1 className="font-display text-4xl font-bold text-gray-900 mb-3">Talk to the platform like a teammate.</h1>
          <p className="text-slate-600 text-sm md:text-base leading-7">
            The agent can profile a dataset, launch training, compare experiments, deploy a winner, generate prediction payloads, and keep the conversation scoped to the dataset you selected.
          </p>
        </div>
      </div>
      {/* Header */}
      <div className="glass-card rounded-[1.75rem] px-6 py-4 flex items-center justify-between shrink-0 mb-4">
        <div>
          <h2 className="font-display text-2xl font-bold text-gray-900 flex items-center gap-2">
            <span>🤖</span> AI Agent
          </h2>
          <p className="text-xs text-gray-500 mt-1">
            Your conversational ML co-pilot
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Dataset selector */}
          <select
            value={activeDataset || ""}
            onChange={(e) => setActiveDataset(e.target.value || undefined)}
            className="text-sm border border-gray-200 rounded-2xl px-3 py-2 bg-white text-gray-700"
          >
            <option value="">No dataset</option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.name}
              </option>
            ))}
          </select>
          <button
            onClick={handleReset}
            className="text-xs text-gray-500 hover:text-red-600 px-3 py-2 rounded-2xl border border-gray-200 hover:border-red-200"
          >
            Reset Chat
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="glass-card rounded-[2rem] flex-1 overflow-y-auto p-6 space-y-4 min-h-[42rem]">
        {messages.length === 0 && (
          <EmptyState
            activeDataset={activeDsName}
            onQuickAction={(text) => {
              setInput(text);
              inputRef.current?.focus();
            }}
          />
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}

        {sending && (
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-sm shrink-0">
              🤖
            </div>
            <div className="bg-slate-100 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="glass-card rounded-[1.75rem] border-t border-gray-200 p-4 shrink-0 mt-4">
        {activeDataset && (
          <div className="text-xs text-gray-400 mb-2 flex items-center gap-1.5">
            <span className="w-2 h-2 bg-emerald-400 rounded-full" />
            Active dataset: {activeDsName}
          </div>
        )}
        <div className="flex items-end gap-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeDataset
                ? "Ask me to analyze, train, compare, deploy, predict, or improve..."
                : "Select a dataset first, or ask me anything..."
            }
            rows={1}
            className="flex-1 resize-none border border-gray-200 rounded-2xl px-4 py-3 text-sm focus:outline-none focus:border-sky-400 focus:ring-1 focus:ring-sky-100 max-h-32 bg-white/90"
            style={{ minHeight: "42px" }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || sending}
            className="bg-sky-600 text-white rounded-2xl px-5 py-3 text-sm font-medium hover:bg-sky-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex items-start gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm shrink-0 ${
          isUser ? "bg-gray-200" : "bg-blue-100"
        }`}
      >
        {isUser ? "👤" : "🤖"}
      </div>
      <div className={`max-w-[75%] space-y-2`}>
        <div
          className={`rounded-[1.4rem] px-4 py-3 text-sm leading-relaxed shadow-sm ${
            isUser
              ? "bg-sky-600 text-white rounded-tr-sm"
              : "bg-slate-100 text-gray-800 rounded-tl-sm"
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div
              className="agent-markdown"
              dangerouslySetInnerHTML={{
                __html: simpleMarkdown(message.content),
              }}
            />
          )}
        </div>

        {/* Tool calls */}
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div className="space-y-1">
            {message.tool_calls.map((tc, i) => (
              <div
                key={i}
                className="flex items-center gap-2 text-xs text-gray-400 px-2"
              >
                <span className="w-4 h-4 rounded bg-emerald-100 text-emerald-600 flex items-center justify-center text-[10px]">
                  ⚡
                </span>
                <span>
                  Called <strong className="text-gray-500">{tc.tool}</strong>
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({
  activeDataset,
  onQuickAction,
}: {
  activeDataset?: string;
  onQuickAction: (text: string) => void;
}) {
  const suggestions = activeDataset
    ? [
        "Analyze this dataset and tell me what you find",
        "Train all available models and compare results",
        "Deploy the best model",
        "Give me sample test JSON and predict",
        "How can I improve the model performance?",
        "What's the current serving status?",
      ]
    : [
        "What can you help me with?",
        "How does this platform work?",
      ];

  return (
    <div className="flex flex-col items-center justify-center h-full text-center py-16">
      <div className="mb-4 flex h-20 w-20 items-center justify-center rounded-[1.75rem] bg-sky-100 text-5xl">🤖</div>
      <h2 className="font-display text-3xl font-bold text-gray-900 mb-2">
        Hi! I'm your ML co-pilot.
      </h2>
      <p className="text-sm text-gray-500 mb-6 max-w-2xl leading-7">
        {activeDataset
          ? `I can analyze "${activeDataset}", engineer features, train models, deploy, predict, and suggest improvements — all through conversation.`
          : "Select a dataset above, then tell me what you want to build. I'll handle the rest."}
      </p>
      <div className="flex flex-wrap justify-center gap-2 max-w-lg">
        {suggestions.map((s) => (
          <button
            key={s}
            onClick={() => onQuickAction(s)}
            className="text-xs bg-white border border-gray-200 rounded-full px-4 py-2 text-gray-600 hover:border-sky-300 hover:text-sky-600 transition-colors"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

/** Very simple markdown to HTML (no dependency needed) */
function simpleMarkdown(text: string): string {
  if (!text) return "";
  let html = text
    // Escape HTML
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    // Code blocks (triple backticks)
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    // Headers
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    // Bold
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    // Inline code
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // Unordered lists
    .replace(/^[•\-\*] (.+)$/gm, "<li>$1</li>")
    // Ordered lists
    .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
    // Line breaks
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>");

  // Wrap consecutive <li> in <ul>
  html = html.replace(
    /(<li>.*?<\/li>(?:<br>)?)+/g,
    (match) => `<ul>${match.replace(/<br>/g, "")}</ul>`
  );

  return `<p>${html}</p>`;
}
