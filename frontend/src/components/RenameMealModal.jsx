import React, { useState } from "react";
import { apiFetch } from "../lib/api";
import { overlayBg, modalBox, modalHeader, modalTitle, inputStyle, primaryBtn, ghostBtn, errorBanner } from "../styles";
import { Icon, Spin } from "./Icon";

const MAX = 80; // the server's meal-name limit (api_v1.MealLabel)

// Artifact v9 M5/M6: rename one logged meal. Only that day's meal changes; its template keeps its name.
export default function RenameMealModal({ meal, onClose, onRenamed, onGone }) {
  const [label, setLabel] = useState(meal.label);
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(false);
  const trimmed = label.trim();

  const save = async (e) => {
    e.preventDefault();
    if (!trimmed || saving) return;
    setSaving(true); setFailed(false);
    try {
      const r = await apiFetch(`/log/meals/${encodeURIComponent(meal.gid)}`, { method: "PATCH", body: JSON.stringify({ label: trimmed }) });
      onRenamed(r.label);
    } catch (err) {
      if (err.status === 404) return onGone();   // deleted elsewhere (Claude, another device): nothing left to rename
      console.error(err); setFailed(true); setSaving(false);
    }
  };

  return (
    <div style={overlayBg}>
      <form onSubmit={save} style={modalBox}>
        <div style={modalHeader}>
          <div style={modalTitle}><Icon n="edit" size={15} style={{ color: "var(--accent)" }} /> Rename meal</div>
          <button type="button" onClick={onClose} disabled={saving} aria-label="Close" style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted)" }}><Icon n="close" size={16} /></button>
        </div>
        <div>
          <input autoFocus aria-label="Meal name" value={label} maxLength={MAX} onChange={e => setLabel(e.target.value)} style={{ ...inputStyle, fontWeight: 700 }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--muted)", marginTop: 6 }}>
            <span>Only this day's meal changes.</span>
            <span>{label.length} / {MAX}</span>
          </div>
        </div>
        {failed && <div role="alert" style={errorBanner}><b>Couldn't rename.</b> Check your connection and try again.</div>}
        <div style={{ display: "flex", gap: 8 }}>
          <button type="button" onClick={onClose} disabled={saving} style={{ ...ghostBtn, flex: 1, opacity: saving ? 0.45 : 1 }}>Cancel</button>
          <button type="submit" disabled={!trimmed || saving} style={{ ...primaryBtn, flex: 1, opacity: !trimmed || saving ? 0.45 : 1 }}>
            {saving ? <Spin size={14} /> : <Icon n="check" size={14} />}
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </form>
    </div>
  );
}
