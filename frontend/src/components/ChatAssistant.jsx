import React, { useState, useRef, useEffect } from "react";
import { apiFetch } from "../lib/api";
import { Icon, Spin } from "./Icon";

export default function ChatAssistant({ open, onOpenChange }) {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hey! What would you like to know about your nutrition today?", greeting: true }
  ]);
  const [input,    setInput]   = useState("");
  const [loading,  setLoading] = useState(false);
  const [error,    setError]   = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length, open]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setError(null);
    setMessages(prev => [...prev, { role: "user", text }]);
    setLoading(true);
    try {
      const data = await apiFetch("/chat", { method: "POST", body: JSON.stringify({ message: text, history: messages.filter(m => !m.greeting) }) });
      setMessages(prev => [...prev, { role: "assistant", text: data.reply }]);
    } catch (e) {
      setError(e.message || "Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;

  return (
    <div style={{
      position: "fixed", bottom: "calc(80px + env(safe-area-inset-bottom, 0px))", right: 16, left: 16, zIndex: 60,
      height: 460,
      background: "var(--surface)", borderRadius: 20,
      border: "1px solid var(--border)",
      boxShadow: "0 12px 48px rgba(0,0,0,0.18)",
      display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      <div style={{ background: "var(--teal)", padding: "12px 16px", display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
        <div style={{ width: 32, height: 32, borderRadius: 10, background: "rgba(174,246,199,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Icon n="nutrition" size={17} style={{ color: "var(--mint)" }} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "white" }}>Nutrition Assistant</div>
          <div style={{ fontSize: 10, color: "rgba(255,255,255,0.65)" }}>Knows your log, goals & 7-day trends</div>
        </div>
        <button onClick={() => onOpenChange(false)} style={{ background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", padding: 4, borderRadius: 6, color: "rgba(255,255,255,0.75)" }}>
          <Icon n="close" size={18} />
        </button>
      </div>

      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "14px", display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: 10, color: "var(--muted)", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 20, padding: "3px 10px", letterSpacing: "0.2px" }}>
            ✦ Fresh start every session — chats don't stick around
          </span>
        </div>

        {messages.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === "user" ? "flex-end" : "flex-start",
            maxWidth: "86%",
            background: m.role === "user" ? "var(--teal)" : "var(--bg)",
            color: m.role === "user" ? "white" : "var(--text)",
            borderRadius: m.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
            padding: "9px 13px",
            fontSize: 13, lineHeight: 1.5,
            boxShadow: "0 1px 3px rgba(0,0,0,0.07)",
          }}>
            {m.text}
          </div>
        ))}
        {loading && (
          <div style={{ alignSelf: "flex-start", background: "var(--bg)", borderRadius: "16px 16px 16px 4px", padding: "10px 14px" }}>
            <Spin size={14} />
          </div>
        )}
        {error && (
          <div style={{ fontSize: 11, color: "var(--danger)", textAlign: "center", padding: "4px 8px" }}>{error}</div>
        )}
      </div>

      <div style={{ padding: "10px 12px", borderTop: "1px solid var(--border)", display: "flex", gap: 8, flexShrink: 0 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          placeholder="Ask about your nutrition…"
          style={{ flex: 1, padding: "8px 12px", borderRadius: 10, border: "1.5px solid var(--border)", fontSize: 16, background: "var(--surface)", color: "var(--text)", outline: "none" }}
        />
        <button onClick={send} disabled={!input.trim() || loading} style={{
          width: 38, height: 38, borderRadius: 10, flexShrink: 0,
          background: input.trim() && !loading ? "var(--teal)" : "var(--border)",
          border: "none", cursor: input.trim() && !loading ? "pointer" : "default",
          display: "flex", alignItems: "center", justifyContent: "center",
          transition: "background 0.15s",
        }}>
          <Icon n="send" size={16} style={{ color: "white" }} />
        </button>
      </div>
    </div>
  );
}
