import React, { useState, useRef, useEffect } from "react";
import { apiFetch } from "../lib/api";
import { Icon, Spin } from "./Icon";

export default function ChatAssistant({ hidden }) {
  const [open,     setOpen]    = useState(false);
  const [messages, setMessages] = useState([]);
  const [input,    setInput]   = useState("");
  const [loading,  setLoading] = useState(false);
  const [error,    setError]   = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setError(null);
    setMessages(prev => [...prev, { role: "user", text }]);
    setLoading(true);
    try {
      const data = await apiFetch("/chat", { method: "POST", body: JSON.stringify({ message: text }) });
      setMessages(prev => [...prev, { role: "assistant", text: data.reply }]);
    } catch (e) {
      setError(e.message || "Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  if (hidden) return null;

  return (
    <>
      <button onClick={() => setOpen(o => !o)} style={{
        position: "fixed", bottom: "calc(90px + env(safe-area-inset-bottom, 0px))", right: 20, zIndex: 60,
        width: 50, height: 50, borderRadius: "50%",
        background: "var(--teal)", border: "none", cursor: "pointer",
        display: "flex", alignItems: "center", justifyContent: "center",
        boxShadow: "0 4px 20px rgba(0,109,119,0.45)",
      }}>
        <Icon n="nutrition" size={22} style={{ color: "white" }} />
        <div style={{
          position: "absolute", top: 0, right: 0,
          width: 18, height: 18, borderRadius: "50%",
          background: "var(--mint)", border: "2px solid white",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Icon n="auto_awesome" size={10} style={{ color: "var(--teal-dk)" }} />
        </div>
      </button>

      {open && (
        <div style={{
          position: "fixed", bottom: "calc(154px + env(safe-area-inset-bottom, 0px))", right: 20, zIndex: 60,
          width: 328, height: 460,
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
              <div style={{ fontSize: 10, color: "rgba(255,255,255,0.65)" }}>Knows your today's macros</div>
            </div>
            <button onClick={() => setOpen(false)} style={{ background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", padding: 4, borderRadius: 6, color: "rgba(255,255,255,0.75)" }}>
              <Icon n="close" size={18} />
            </button>
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "14px", display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ fontSize: 10, color: "var(--muted)", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 20, padding: "3px 10px", letterSpacing: "0.2px" }}>
                ✦ Fresh start every session — chats don't stick around
              </span>
            </div>
            {messages.length === 0 && (
              <div style={{ color: "var(--muted)", fontSize: 12, textAlign: "center", lineHeight: 1.6, marginTop: 8 }}>
                Ask me about your macros, food choices,<br />or how to hit your goals today.
              </div>
            )}
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
            <div ref={bottomRef} />
          </div>

          <div style={{ padding: "10px 12px", borderTop: "1px solid var(--border)", display: "flex", gap: 8, flexShrink: 0 }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Ask about your nutrition…"
              style={{ flex: 1, padding: "8px 12px", borderRadius: 10, border: "1.5px solid var(--border)", fontSize: 13, background: "var(--surface)", color: "var(--text)", outline: "none" }}
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
      )}
    </>
  );
}
