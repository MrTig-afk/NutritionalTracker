import React, { useState } from "react";
import { apiFetch } from "../lib/api";
import { deletionDate } from "../lib/nutrition";
import { Spin } from "./Icon";

// L2: on every tab while the account is inside its 15 days. L4: Keep my account.
export default function DeletionBanner({ deleteAfter, onKept, onExport }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const keep = async () => {
    setBusy(true); setError(null);
    try {
      const r = await apiFetch("/account/restore", { method: "POST" });
      // Not restored here: kept already on another device, or the date has passed. The server knows which.
      if (r.restored || !(await apiFetch("/account/deletion")).delete_after) onKept();
      else setError("Couldn't keep your account. Try again.");
    } catch (e) {
      // reauth_required carries its own instruction (sign out and back in first)
      setError(e.errorType === "reauth_required" ? e.message : "Couldn't keep your account. Try again.");
    } finally { setBusy(false); }
  };

  return (
    <div role="status" style={{ background: "var(--orange-lt)", color: "var(--orange)", padding: "10px 16px" }}>
      <div style={{ maxWidth: 560, margin: "0 auto", display: "flex", flexDirection: "column", gap: 8 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 800 }}>Deleting on {deletionDate(deleteAfter)}</div>
          <div style={{ fontSize: 12 }}>You can still view and export everything.</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={keep} disabled={busy}
            style={{ flex: 1, padding: "8px 10px", borderRadius: 10, border: "none", background: "var(--teal)", color: "#fff", fontSize: 13, fontWeight: 700, cursor: busy ? "not-allowed" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
            {busy ? <Spin size={14} color="white" /> : "Keep my account"}
          </button>
          <button onClick={onExport}
            style={{ flex: 1, padding: "8px 10px", borderRadius: 10, border: "1px solid var(--orange)", background: "transparent", color: "var(--orange)", fontSize: 13, fontWeight: 700, cursor: "pointer" }}>
            Export
          </button>
        </div>
        {error && <div style={{ fontSize: 12, fontWeight: 700 }}>{error}</div>}
      </div>
    </div>
  );
}

export function KeptNotice() {
  return (
    <div role="status" style={{ background: "var(--mint)", color: "#0B3D22", padding: "10px 16px", fontSize: 13, fontWeight: 700, textAlign: "center" }}>
      Your account is no longer being deleted. Everything is back.
    </div>
  );
}
