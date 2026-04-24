import React from "react";

export const PALETTE_CSS = `
  @keyframes ns-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
  :root {
    --teal:      #006D77;
    --teal-lt:   #DCF4F5;
    --teal-md:   #004E56;
    --teal-dk:   #003940;
    --mint:      #AEF6C7;
    --mint-dk:   #1A6B3C;
    --bg:        #F4FAF9;
    --surface:   #FFFFFF;
    --off:       #EFF5F5;
    --off2:      #E0ECEB;
    --white:     #FFFFFF;
    --border:    #C0CBCA;
    --border2:   #8FA09F;
    --text:      #191C1C;
    --text2:     #3F4949;
    --muted:     #6F7979;
    --orange:    #C25700;
    --orange-lt: #FFE0C7;
    --purple:    #6B5EA8;
    --purp-lt:   #EEEDF6;
    --brown:     #4A3728;
    --brown-lt:  #F5EDE0;
    --danger:    #BA1A1A;
    --danger-lt: #FFDAD6;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: 'Manrope', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
  }
  input, select, textarea, button { font-family: inherit; }
  input:focus, select:focus { outline: 2px solid var(--teal); outline-offset: -1px; }

  /* Responsive nav */
  .ns-bottom-nav { display: flex; }
  .ns-top-tabs   { display: none; }
  .ns-content    { padding-bottom: 96px; }

  .ns-scan-grid    { grid-template-columns: 1fr; }
  .ns-tracker-grid { display: flex; flex-direction: column; gap: 16px; }

  @media (min-width: 768px) {
    .ns-bottom-nav   { display: none; }
    .ns-top-tabs     { display: flex; }
    .ns-content      { max-width: 100% !important; padding-bottom: 32px; }
    .ns-scan-grid    { grid-template-columns: 1fr 1fr !important; }
    .ns-tracker-grid { display: grid !important; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
  }
`;

export const card       = { background: "var(--surface)", borderRadius: 16, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.06), 0 0 0 1px var(--border)" };
export const cardHeader  = { background: "var(--off)", padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" };
export const inputStyle  = { width: "100%", padding: "10px 13px", background: "var(--surface)", border: "1.5px solid var(--border)", borderRadius: 10, fontSize: 14, color: "var(--text)" };
export const labelStyle  = { fontSize: 11, color: "var(--muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.6px", display: "block", marginBottom: 5 };
export const primaryBtn  = { width: "100%", padding: "13px", background: "var(--teal)", color: "white", border: "none", borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 };
export const mintBtn     = { width: "100%", padding: "13px", background: "var(--mint)", color: "var(--mint-dk)", border: "none", borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 };
export const ghostBtn    = { padding: "9px 14px", background: "var(--off)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12, fontWeight: 600, color: "var(--text2)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 };
export const overlayBg   = { position: "fixed", inset: 0, zIndex: 50, background: "rgba(0,0,0,0.55)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", padding: 16 };
export const modalBox    = { width: "100%", maxWidth: 420, background: "var(--surface)", borderRadius: 20, padding: 24, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 20px 60px rgba(0,0,0,0.2)" };
export const modalHeader = { display: "flex", alignItems: "center", justifyContent: "space-between" };
export const modalTitle  = { fontSize: 15, fontWeight: 700, color: "var(--text)", display: "flex", alignItems: "center", gap: 8 };
export const pillRow     = { display: "flex", padding: 4, background: "var(--off2)", borderRadius: 10, gap: 3 };

export const macroCells = (vals) => vals.map(({ label, value, unit, color }) => (
  <div key={label} style={{ flex: 1, background: "var(--off)", border: "1px solid var(--border)", borderRadius: 10, padding: "8px 6px", textAlign: "center" }}>
    <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.3px" }}>{label}</div>
    <div style={{ fontSize: 16, fontWeight: 800, color, marginTop: 2 }}>{value.toFixed(1)}</div>
    <div style={{ fontSize: 10, color: "var(--muted)" }}>{unit}</div>
  </div>
));
