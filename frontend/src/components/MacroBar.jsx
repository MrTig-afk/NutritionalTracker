import React from "react";

export default function MacroBar({ label, current, goal, color }) {
  const pct = goal > 0 ? Math.min(100, (current / goal) * 100) : 0;
  const over = current > goal;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
        <span style={{ color: "var(--muted)", fontWeight: 600 }}>{label}</span>
        <span style={{ color: over ? "var(--danger)" : "var(--text)", fontWeight: 600 }}>{current.toFixed(1)} / {goal}{over && " ⚠"}</span>
      </div>
      <div style={{ height: 7, background: "var(--off2)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ height: "100%", borderRadius: 4, background: over ? "var(--danger)" : color, width: `${pct}%`, transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
}
