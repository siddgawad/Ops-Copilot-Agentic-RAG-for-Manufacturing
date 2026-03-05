"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Sparkles, RotateCcw, Factory } from "lucide-react";

type Source = {
  text: string;
  source: string;
  score: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm Ops Copilot. I'm connected to 500+ pages of Fanuc robot manuals and manufacturing SOPs. How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "waking" | "offline">("checking");
  const [chunksIndexed, setChunksIndexed] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Use env var in production, default to local FastAPI in dev
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Health check on page load — detect cold starts
  useEffect(() => {
    const checkHealth = async () => {
      setServerStatus("checking");
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 90000); // 90s timeout for cold start
        setServerStatus("waking");
        const res = await fetch(`${API_URL}/health`, { signal: controller.signal });
        clearTimeout(timeout);
        if (res.ok) {
          const data = await res.json();
          setChunksIndexed(data.chunks_indexed || 0);
          setServerStatus("online");
        } else {
          setServerStatus("offline");
        }
      } catch {
        setServerStatus("offline");
      }
    };
    checkHealth();
  }, [API_URL]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      // Build history payload (last 3 turns = 6 messages)
      const history = messages
        .filter(m => m.role !== "assistant" || m.content !== "Hello! I'm Ops Copilot. I'm connected to 500+ pages of Fanuc robot manuals and manufacturing SOPs. How can I help you today?")
        .slice(-6)
        .map((m, i, arr) => {
          if (m.role === "user" && arr[i + 1]?.role === "assistant") {
            return { question: m.content, answer: arr[i + 1].content };
          }
          return null;
        }).filter(Boolean);

      // 90s timeout to handle Render free tier cold starts
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 90000);

      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage,
          history: history
        }),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!response.ok) throw new Error("API request failed");

      const data = await response.json();
      setServerStatus("online");

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
    } catch (error) {
      console.error(error);
      const errorMsg = serverStatus !== "online"
        ? "⏳ The backend server is waking up (free tier spins down after inactivity). Please wait ~60 seconds and try again."
        : "⚠️ Error connecting to the RAG backend. The server may have gone to sleep — please retry in a moment.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: errorMsg },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex h-screen w-full flex-col bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-zinc-800 bg-zinc-950/50 px-6 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-500/10 text-orange-500">
            <Factory size={18} />
          </div>
          <div>
            <h1 className="font-semibold tracking-tight text-zinc-100">Ops Copilot</h1>
            <p className="text-xs text-zinc-400">Hybrid RAG System</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs text-zinc-400">
            {serverStatus === "online" ? (
              <>
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                <span className="hidden sm:inline">{chunksIndexed} chunks indexed</span>
              </>
            ) : serverStatus === "waking" || serverStatus === "checking" ? (
              <>
                <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
                <span className="hidden sm:inline">Server waking up...</span>
              </>
            ) : (
              <>
                <span className="h-2 w-2 rounded-full bg-red-500" />
                <span className="hidden sm:inline">Server offline</span>
              </>
            )}
          </div>
          <button
            onClick={() => setMessages([messages[0]])}
            className="flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 transition-colors"
          >
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-6 pb-4">
          {messages.map((message, i) => (
            <div
              key={i}
              className={`flex gap-4 ${message.role === "user" ? "flex-row-reverse" : "flex-row"
                }`}
            >
              {/* Avatar */}
              <div
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${message.role === "assistant"
                  ? "bg-orange-500/20 text-orange-500"
                  : "bg-zinc-800 text-zinc-300"
                  }`}
              >
                {message.role === "assistant" ? <Sparkles size={16} /> : "U"}
              </div>

              {/* Message Content */}
              <div
                className={`flex flex-col gap-2 max-w-[85%] ${message.role === "user" ? "items-end" : "items-start"
                  }`}
              >
                <div
                  className={`rounded-2xl px-4 py-3 shadow-sm ${message.role === "user"
                    ? "bg-zinc-800 text-zinc-100 rounded-tr-sm"
                    : "bg-zinc-900 border border-zinc-800 text-zinc-300 rounded-tl-sm prose prose-invert prose-sm"
                    }`}
                >
                  <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
                </div>

                {/* Citations */}
                {message.sources && message.sources.length > 0 && (
                  <div className="w-full mt-2 space-y-2">
                    <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider pl-1">
                      Retrieved Sources
                    </p>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {message.sources.map((src, idx) => (
                        <div
                          key={idx}
                          className="rounded-lg border border-zinc-800/60 bg-zinc-900/50 p-3 text-xs transition-colors hover:border-zinc-700"
                        >
                          <div className="mb-1 flex items-center justify-between gap-2">
                            <span className="font-semibold text-orange-400/90 truncate">
                              {src.source}
                            </span>
                            <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
                              score: {src.score.toFixed(3)}
                            </span>
                          </div>
                          <p className="line-clamp-3 text-zinc-500 leading-relaxed">
                            &ldquo;{src.text}&rdquo;
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-orange-500/20 text-orange-500">
                <Sparkles size={16} className="animate-pulse" />
              </div>
              <div className="rounded-2xl bg-zinc-900 border border-zinc-800 px-4 py-3 rounded-tl-sm">
                <div className="flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-orange-500/60" style={{ animationDelay: "0ms" }}></span>
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-orange-500/60" style={{ animationDelay: "150ms" }}></span>
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-orange-500/60" style={{ animationDelay: "300ms" }}></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-zinc-800 bg-zinc-950 p-4 sm:p-6 pb-6">
        <div className="mx-auto max-w-3xl">
          <form
            onSubmit={handleSubmit}
            className="flex items-end gap-2 rounded-xl border border-zinc-800 bg-zinc-900 p-2 shadow-sm focus-within:border-zinc-700 focus-within:ring-1 focus-within:ring-zinc-700 transition-all"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder="Ask about Fanuc operations, e-stops, or spindle tolerances..."
              className="max-h-32 min-h-12 w-full resize-none bg-transparent px-3 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:outline-none"
              rows={1}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-orange-600 text-white transition-colors hover:bg-orange-500 disabled:bg-zinc-800 disabled:text-zinc-600 mb-1 mr-1"
            >
              <Send size={18} className={input.trim() && !isLoading ? "-ml-0.5 mt-0.5" : ""} />
            </button>
          </form>
          <div className="mt-2 text-center text-xs text-zinc-600">
            Powered by FastAPI, ChromaDB, BM25, and GPT-4o-mini
          </div>
        </div>
      </div>
    </main>
  );
}
