import React, { useState } from "react";
import { NUTRIENT_META, getFallbackMeta, parseNumeric, extractServingGrams } from "../lib/nutrition";
import { inputStyle } from "../styles";
import { Icon } from "./Icon";

export default function NutrientGrid({ data, activeTab, per100gData }) {
  const [customGrams, setCustomGrams] = useState("");
  if (!data || Object.keys(data).length === 0) {
    return <div style={{ padding: "48px 16px", textAlign: "center", border: "2px dashed var(--border)", borderRadius: 16, color: "var(--muted)", fontSize: 13, fontStyle: "italic" }}>No data extracted for this view</div>;
  }
  const customG = parseFloat(customGrams); const isValid = !isNaN(customG) && customG > 0;
  let baseGrams = null, scaleFrom = data, baseLabel = null, warnMsg = null;
  if (activeTab === "per_100g") { baseGrams = 100; baseLabel = "100g"; }
  else {
    const sg = extractServingGrams(data.size);
    if (sg) { baseGrams = sg; baseLabel = `${data.size} (${sg}g)`; }
    else if (per100gData && Object.keys(per100gData).length > 0) { baseGrams = 100; scaleFrom = per100gData; baseLabel = "100g (fallback)"; warnMsg = "No serving size — scaling from per 100g"; }
    else { warnMsg = "Cannot scale — no size or per 100g data"; }
  }
  const scalingActive = isValid && baseGrams !== null;
  const factor = scalingActive ? customG / baseGrams : null;
  const getDisplay = (key, rawValue) => {
    if (!scalingActive) return { display: rawValue, adjusted: false };
    const base = parseNumeric(scaleFrom[key] ?? rawValue);
    if (base === null) return { display: rawValue, adjusted: false };
    return { display: parseFloat((base * factor).toFixed(2)), adjusted: true, baseDisplay: rawValue };
  };
  const entries = Object.entries(data).filter(([k]) => k !== "size");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ background: "var(--off)", border: "1px solid var(--border)", borderRadius: 12, padding: "14px 16px" }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--brown)", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
          <Icon n="scale" size={13} style={{ color: "var(--teal)" }} /> Custom Serving Calculator
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <input type="number" min="1" step="any" placeholder="e.g. 58" value={customGrams} onChange={e => setCustomGrams(e.target.value)} style={{ ...inputStyle, flex: 1 }} />
          <span style={{ fontSize: 13, fontWeight: 700, color: "var(--muted)" }}>g</span>
          {customGrams && <button onClick={() => setCustomGrams("")} style={{ fontSize: 11, color: "var(--muted)", background: "none", border: "none", cursor: "pointer" }}>Clear</button>}
        </div>
        <p style={{ fontSize: 11, color: warnMsg ? "var(--orange)" : "var(--muted)", marginTop: 8 }}>
          {warnMsg || `Base: ${baseLabel}${scalingActive ? ` → ${customG}g · ×${factor.toFixed(4)}` : ""}`}
        </p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, background: "var(--border)", borderRadius: 12, overflow: "hidden" }}>
        {entries.map(([key, value]) => {
          const meta = NUTRIENT_META[key] ?? getFallbackMeta(key);
          const { display, adjusted, baseDisplay } = getDisplay(key, value);
          return (
            <div key={key} style={{ background: adjusted ? "var(--teal-lt)" : "var(--white)", padding: "12px 14px", display: "flex", flexDirection: "column", gap: 4 }}>
              <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", lineHeight: 1.3 }}>{meta.label}</div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 3 }}>
                <span style={{ fontSize: 20, fontWeight: 800, color: meta.color, lineHeight: 1.1 }}>{display}</span>
                {meta.unit && <span style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600 }}>{meta.unit}</span>}
                {adjusted && <span style={{ fontSize: 10, color: "var(--teal)", marginLeft: 2 }}>adj.</span>}
              </div>
              {adjusted && <div style={{ fontSize: 10, color: "var(--muted)" }}>base: {baseDisplay}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
