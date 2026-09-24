import React from "react";

export const Icon = ({ n, size = 20, style: s = {}, cls = "" }) => (
  <span
    aria-hidden="true"
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

// The Claude connector spark (design/userflow.artifact.html "H4 · Allow"): 8 long
// + 8 short petals and a centre circle. Shared by TrackerTab's ViaClaude chip and
// AllowPage; lane H's Connected apps list (a later batch) will use it too.
export const Spark = ({ size = 10, color = "currentColor" }) => {
  const long = [0, 45, 90, 135, 180, 225, 270, 315];
  const short = [22, 67, 112, 157, 202, 247, 292, 337];
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill={color} aria-hidden="true">
      {long.map(a => <path key={`l${a}`} d="M12 12 L10.3 3.4 Q12 1.2 13.7 3.4 Z" transform={`rotate(${a} 12 12)`} />)}
      {short.map(a => <path key={`s${a}`} d="M12 12 L10.9 5.6 Q12 4.4 13.1 5.6 Z" transform={`rotate(${a} 12 12)`} />)}
      <circle cx="12" cy="12" r="1.6" />
    </svg>
  );
};
