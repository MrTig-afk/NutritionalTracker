import React, { useEffect, useState } from "react";
import { overlayBg, modalBox } from "../styles";
import { _register } from "../lib/confirm";

// Styled like the Settings "Delete account?" modal. Mounted once in App;
// call sites use `confirm()` from lib/confirm.js.
export default function ConfirmHost() {
  const [queue, setQueue] = useState([]);          // concurrent confirm() calls wait their turn
  const req = queue[0] || null;
  const done = v => { req?.resolve(v); setQueue(q => q.slice(1)); };
  useEffect(() => {
    const pending = [];
    _register(r => { pending.push(r); setQueue(q => [...q, r]); });
    return () => { _register(null); pending.forEach(r => r.resolve(false)); };  // unmount: nothing hangs
  }, []);
  useEffect(() => {
    if (!req) return;
    const onKey = e => { if (e.key === "Escape") done(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  });
  if (!req) return null;
  return (
    <div style={{ ...overlayBg, zIndex: 90 }} onClick={() => req.cancel && done(false)}>
      <div role="alertdialog" aria-modal="true" aria-label={req.title} onClick={e => e.stopPropagation()} style={{ ...modalBox, gap: 12 }}>
        <div style={{ fontSize: 17, fontWeight: 800, color: "var(--danger)" }}>{req.title}</div>
        <div style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.6 }}>{req.message}</div>
        <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
          {req.cancel && (
            <button onClick={() => done(false)}
              style={{ flex: 1, padding: "12px", background: "var(--off)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 13, fontWeight: 700, cursor: "pointer" }}>
              Cancel
            </button>
          )}
          <button autoFocus onClick={() => done(true)}
            style={{ flex: 1, padding: "12px", background: "var(--danger)", color: "var(--on-danger)", border: "none", borderRadius: 10, fontSize: 13, fontWeight: 700, cursor: "pointer" }}>
            {req.okLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
