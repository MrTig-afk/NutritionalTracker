import React, { useState } from "react";
import { apiFetch } from "../lib/api";
import { resolveNutrition, extractServingGrams, parseNumeric } from "../lib/nutrition";
import { overlayBg, modalBox, modalHeader, modalTitle, inputStyle, labelStyle, mintBtn, pillRow, macroCells } from "../styles";
import { Icon, Spin } from "./Icon";

export default function AddToLogModal({ item, onClose, onAdded }) {
  const [mode, setMode] = useState("serving");
  const [servings, setServings] = useState("1");
  const [grams, setGrams] = useState("");
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);
  const [manualName, setManualName] = useState(item.name || "");
  const [manualCal, setManualCal] = useState("");
  const [manualProt, setManualProt] = useState("");
  const [manualCarb, setManualCarb] = useState("");
  const [manualFat, setManualFat] = useState("");
  const [manualFibre, setManualFibre] = useState("");

  const rawNutrition = item.nutrition || {};
  const per100g    = rawNutrition.per_100g    && Object.keys(rawNutrition.per_100g).length    > 0 ? rawNutrition.per_100g    : null;
  const perServing = rawNutrition.per_serving && Object.keys(rawNutrition.per_serving).length > 0 ? rawNutrition.per_serving : null;
  const baseNutrition = perServing ?? per100g ?? resolveNutrition(rawNutrition);
  const servingGrams = perServing ? extractServingGrams(perServing.size) : null;
  const servingsNum = parseFloat(servings) || 1; const gramsNum = parseFloat(grams) || 0;

  let scaledNutrition = { ...baseNutrition }; let scalingInfo = null;
  if (mode === "serving") {
    scalingInfo = { factor: servingsNum, baseLabel: perServing ? `per serving${perServing.size ? ` (${perServing.size})` : ""}` : "per 100g" };
  } else if (mode === "grams") {
    if (per100g && gramsNum > 0) { scalingInfo = { factor: gramsNum / 100, baseLabel: "per 100g", targetLabel: `${gramsNum}g` }; scaledNutrition = per100g; }
    else if (perServing && servingGrams && gramsNum > 0) { scalingInfo = { factor: gramsNum / servingGrams, baseLabel: `per serving (${servingGrams}g)`, targetLabel: `${gramsNum}g` }; scaledNutrition = perServing; }
    else { scalingInfo = { factor: 1, baseLabel: "no size data", targetLabel: `${gramsNum}g`, warn: true }; }
  }

  const factor = scalingInfo?.factor ?? 1;
  const getVal = (key) => (parseNumeric(scaledNutrition[key]) || 0) * factor;
  const cal  = mode === "manual" ? (parseFloat(manualCal)   || 0) : getVal("calories");
  const prot = mode === "manual" ? (parseFloat(manualProt)  || 0) : getVal("protein");
  const carb = mode === "manual" ? (parseFloat(manualCarb)  || 0) : getVal("carbohydrates");
  const fat  = mode === "manual" ? (parseFloat(manualFat)   || 0) : getVal("fat");
  const fib  = mode === "manual" ? (parseFloat(manualFibre) || 0) : getVal("fibre");

  const buildSubmitNutrition = () => {
    if (mode === "manual") return { per_serving: { size: "1 serving", calories: Math.round(cal), protein: `${prot.toFixed(1)}g`, carbohydrates: `${carb.toFixed(1)}g`, fat: `${fat.toFixed(1)}g`, fibre: `${fib.toFixed(1)}g` } };
    if (mode === "serving") return rawNutrition;
    return { per_serving: { size: `${gramsNum}g`, calories: Math.round(cal), fat: `${fat.toFixed(1)}g`, carbohydrates: `${carb.toFixed(1)}g`, protein: `${prot.toFixed(1)}g`, fibre: `${fib.toFixed(1)}g`, ...(scaledNutrition.saturated_fat ? { saturated_fat: `${(parseNumeric(scaledNutrition.saturated_fat)||0)*factor}g` } : {}), ...(scaledNutrition.sugars ? { sugars: `${(parseNumeric(scaledNutrition.sugars)||0)*factor}g` } : {}), ...(scaledNutrition.sodium ? { sodium: `${(parseNumeric(scaledNutrition.sodium)||0)*factor}g` } : {}) } };
  };

  const canSave = mode === "serving" ? servingsNum > 0 : mode === "grams" ? gramsNum > 0 : manualCal !== "" || manualProt !== "" || manualCarb !== "" || manualFat !== "";

  const save = async () => {
    if (!canSave) return; setSaving(true);
    try {
      const submitServings = mode === "serving" ? servingsNum : 1;
      const submitName = manualName.trim() || item.name;
      await apiFetch("/log", { method: "POST", body: JSON.stringify({ name: submitName, servings: submitServings, nutrition: buildSubmitNutrition() }) });
      setStatus({ type: "ok", msg: "Added to today's log!" }); onAdded(); setTimeout(() => { onClose(); }, 900);
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
    finally { setSaving(false); }
  };

  const activePill   = { flex: 1, padding: "7px", background: "var(--teal)", color: "white", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: "pointer" };
  const inactivePill = { flex: 1, padding: "7px", background: "transparent", color: "var(--muted)", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer" };

  return (
    <div style={overlayBg}>
      <div style={modalBox}>
        <div style={modalHeader}>
          <div style={modalTitle}><Icon n="calendar_today" size={15} style={{ color: "var(--mint-dk)" }} /> Add to Today's Log</div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted)" }}><Icon n="close" size={16} /></button>
        </div>
        <input value={manualName} onChange={e => setManualName(e.target.value)} placeholder="Item name..." style={{ ...inputStyle, fontWeight: 700 }} />
        <div style={pillRow}>
          {[["serving", "Per Serving"], ["grams", "By Weight"], ["manual", "Manual"]].map(([m, label]) => (
            <button key={m} onClick={() => setMode(m)} style={mode === m ? activePill : inactivePill}>{label}</button>
          ))}
        </div>
        {mode === "serving" && (
          <div>
            <label style={labelStyle}>Servings</label>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button onClick={() => setServings(s => String(Math.max(0.5, parseFloat(s) - 0.5)))} style={{ width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="remove" size={12} /></button>
              <input type="number" min="0.5" step="0.5" value={servings} onChange={e => setServings(e.target.value)} style={{ ...inputStyle, width: 80, textAlign: "center" }} />
              <button onClick={() => setServings(s => String(parseFloat(s) + 0.5))} style={{ width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="add" size={12} /></button>
            </div>
            {baseNutrition.size && <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>Base: {baseNutrition.size}</p>}
          </div>
        )}
        {mode === "grams" && (
          <div>
            <label style={labelStyle}>Amount (grams)</label>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <input type="number" min="1" step="1" value={grams} onChange={e => setGrams(e.target.value)} placeholder="e.g. 150" style={{ ...inputStyle, flex: 1 }} />
              <span style={{ fontSize: 14, fontWeight: 700, color: "var(--muted)" }}>g</span>
            </div>
          </div>
        )}
        {mode === "manual" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {[["Calories (kcal)", manualCal, setManualCal], ["Protein (g)", manualProt, setManualProt], ["Carbs (g)", manualCarb, setManualCarb], ["Fat (g)", manualFat, setManualFat], ["Fibre (g)", manualFibre, setManualFibre]].map(([label, val, set]) => (
                <div key={label}>
                  <label style={{ ...labelStyle, fontSize: 10 }}>{label}</label>
                  <input type="number" min="0" step="any" value={val} onChange={e => set(e.target.value)} placeholder="0" style={inputStyle} />
                </div>
              ))}
            </div>
          </div>
        )}
        {scalingInfo && mode !== "manual" && (
          <p style={{ fontSize: 11, color: scalingInfo.warn ? "var(--orange)" : "var(--muted)" }}>
            {scalingInfo.warn ? `⚠ No size data — using base values as-is`
              : `Base: ${scalingInfo.baseLabel}${scalingInfo.targetLabel ? ` → ${scalingInfo.targetLabel} ×${scalingInfo.factor.toFixed(3)}` : ""}${mode === "serving" && servingsNum !== 1 ? ` × ${servingsNum} servings` : ""}`}
          </p>
        )}
        <div style={{ display: "flex", gap: 8 }}>
          {macroCells([
            { label: "Cal",   value: cal,  unit: "kcal", color: "var(--orange)"  },
            { label: "Prot",  value: prot, unit: "g",    color: "var(--teal)"    },
            { label: "Carbs", value: carb, unit: "g",    color: "var(--purple)"  },
            { label: "Fat",   value: fat,  unit: "g",    color: "var(--brown)"   },
          ])}
        </div>
        {status && <p style={{ fontSize: 12, color: status.type === "ok" ? "var(--mint-dk)" : "var(--danger)" }}>{status.msg}</p>}
        <button onClick={save} disabled={saving || !canSave} style={{ ...mintBtn, opacity: (saving || !canSave) ? 0.45 : 1 }}>
          {saving ? <Spin size={14} /> : <Icon n="add" size={14} />}
          {saving ? "Adding..." : "Add to Log"}
        </button>
      </div>
    </div>
  );
}
