import React, { useState } from "react";
import { apiFetch } from "../lib/api";
import { extractServingGrams, parseNumeric } from "../lib/nutrition";
import { overlayBg, modalBox, modalHeader, modalTitle, inputStyle, labelStyle, primaryBtn, pillRow, macroCells } from "../styles";
import { Icon, Spin } from "./Icon";

export default function EditLogModal({ entry, onClose, onSaved }) {
  const [mode, setMode] = useState("serving");
  const [servings, setServings] = useState(String(entry.servings || 1));
  const [grams, setGrams] = useState("");
  const [name, setName] = useState(entry.name || "");
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);

  const nutrition = entry.nutrition || {};
  const per100g    = nutrition.per_100g    && Object.keys(nutrition.per_100g).length    > 0 ? nutrition.per_100g    : null;
  const perServing = nutrition.per_serving && Object.keys(nutrition.per_serving).length > 0 ? nutrition.per_serving : null;
  const baseNutrition = perServing ?? per100g ?? nutrition;
  const servingGrams = perServing ? extractServingGrams(perServing.size) : null;
  const servingsNum = parseFloat(servings) || 1; const gramsNum = parseFloat(grams) || 0;

  let scaledNutrition = { ...baseNutrition }; let scalingInfo = null;
  if (mode === "serving") { scalingInfo = { factor: servingsNum, baseLabel: perServing?.size ? `per serving (${perServing.size})` : "per serving" }; }
  else {
    if (per100g && gramsNum > 0) { scalingInfo = { factor: gramsNum / 100, baseLabel: "per 100g", targetLabel: `${gramsNum}g` }; scaledNutrition = per100g; }
    else if (perServing && servingGrams && gramsNum > 0) { scalingInfo = { factor: gramsNum / servingGrams, baseLabel: `per serving (${servingGrams}g)`, targetLabel: `${gramsNum}g` }; scaledNutrition = perServing; }
    else { scalingInfo = { factor: 1, baseLabel: "no size data", warn: true }; }
  }

  const factor = scalingInfo?.factor ?? 1;
  const getVal = (key) => (parseNumeric(scaledNutrition[key]) || 0) * factor;
  const cal = getVal("calories"); const prot = getVal("protein"); const carb = getVal("carbohydrates"); const fat = getVal("fat");

  const buildNutrition = () => {
    if (mode === "serving") return entry.nutrition;
    return { per_serving: { size: `${gramsNum}g`, calories: Math.round(cal), fat: `${fat.toFixed(1)}g`, carbohydrates: `${carb.toFixed(1)}g`, protein: `${prot.toFixed(1)}g`, fibre: `${getVal("fibre").toFixed(1)}g` } };
  };

  const save = async () => {
    if (mode === "serving" && servingsNum <= 0) return; if (mode === "grams" && gramsNum <= 0) return;
    setSaving(true);
    try {
      const submitServings = mode === "serving" ? servingsNum : 1;
      await apiFetch(`/log/${entry.log_id}`, { method: "PUT", body: JSON.stringify({ name: name.trim() || entry.name, servings: submitServings, nutrition: buildNutrition() }) });
      setStatus({ type: "ok", msg: "Updated!" }); onSaved(); setTimeout(onClose, 700);
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
    finally { setSaving(false); }
  };

  const activePill   = { flex: 1, padding: "7px", background: "var(--teal)", color: "white", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: "pointer" };
  const inactivePill = { flex: 1, padding: "7px", background: "transparent", color: "var(--muted)", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer" };

  return (
    <div style={overlayBg}>
      <div style={modalBox}>
        <div style={modalHeader}>
          <div style={modalTitle}><Icon n="edit" size={15} style={{ color: "var(--teal)" }} /> Edit Log Entry</div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted)" }}><Icon n="close" size={16} /></button>
        </div>
        <div>
          <label style={labelStyle}>Name</label>
          <input value={name} onChange={e => setName(e.target.value)} style={{ ...inputStyle, fontWeight: 700 }} />
        </div>
        <div style={pillRow}>
          {[["serving", "Per Serving"], ["grams", "By Weight"]].map(([m, label]) => (
            <button key={m} onClick={() => setMode(m)} style={mode === m ? activePill : inactivePill}>{label}</button>
          ))}
        </div>
        {mode === "serving" ? (
          <div>
            <label style={labelStyle}>Servings</label>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button onClick={() => setServings(s => String(Math.max(0.5, parseFloat(s) - 0.5)))} style={{ width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="remove" size={12} /></button>
              <input type="number" min="0.5" step="0.5" value={servings} onChange={e => setServings(e.target.value)} style={{ ...inputStyle, width: 80, textAlign: "center" }} />
              <button onClick={() => setServings(s => String(parseFloat(s) + 0.5))} style={{ width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="add" size={12} /></button>
            </div>
            {baseNutrition?.size && <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>Base: {baseNutrition.size}</p>}
          </div>
        ) : (
          <div>
            <label style={labelStyle}>Amount (grams)</label>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <input type="number" min="1" step="1" value={grams} onChange={e => setGrams(e.target.value)} placeholder="e.g. 150" style={{ ...inputStyle, flex: 1 }} />
              <span style={{ fontSize: 14, fontWeight: 700, color: "var(--muted)" }}>g</span>
            </div>
          </div>
        )}
        {scalingInfo && <p style={{ fontSize: 11, color: scalingInfo.warn ? "var(--orange)" : "var(--muted)" }}>{scalingInfo.warn ? "⚠ No size data" : `Base: ${scalingInfo.baseLabel}${scalingInfo.targetLabel ? ` → ${scalingInfo.targetLabel}` : ""}${mode === "serving" && servingsNum !== 1 ? ` × ${servingsNum}` : ""}`}</p>}
        <div style={{ display: "flex", gap: 8 }}>
          {macroCells([
            { label: "Cal",   value: cal,  unit: "kcal", color: "var(--orange)" },
            { label: "Prot",  value: prot, unit: "g",    color: "var(--teal)"   },
            { label: "Carbs", value: carb, unit: "g",    color: "var(--purple)" },
            { label: "Fat",   value: fat,  unit: "g",    color: "var(--brown)"  },
          ])}
        </div>
        {status && <p style={{ fontSize: 12, color: status.type === "ok" ? "var(--mint-dk)" : "var(--danger)" }}>{status.msg}</p>}
        <button onClick={save} disabled={saving || (mode === "serving" ? servingsNum <= 0 : gramsNum <= 0)} style={{ ...primaryBtn, opacity: saving ? 0.45 : 1 }}>
          {saving ? <Spin size={14} /> : <Icon n="check" size={14} />}
          {saving ? "Saving..." : "Save Changes"}
        </button>
      </div>
    </div>
  );
}
