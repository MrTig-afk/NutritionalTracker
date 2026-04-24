import React from "react";

export const Icon = ({ n, size = 20, style: s = {}, cls = "" }) => (
  <span
    className={`material-symbols-outlined${cls ? " " + cls : ""}`}
    style={{ fontSize: size, lineHeight: 1, display: "inline-flex", alignItems: "center", userSelect: "none", flexShrink: 0, ...s }}
  >
    {n}
  </span>
);

export const Spin = ({ size = 20, color = "var(--teal)" }) => (
  <div style={{
    width: size, height: size, borderRadius: "50%",
    border: "2.5px solid rgba(0,0,0,0.08)",
    borderTopColor: color,
    animation: "ns-spin 0.75s linear infinite",
    flexShrink: 0, display: "inline-block",
  }} />
);
