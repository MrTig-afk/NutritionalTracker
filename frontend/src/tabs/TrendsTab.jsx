import React, { useState, useEffect } from "react";
import { apiFetch } from "../lib/api";
import { Icon, Spin } from "../components/Icon";

function MacroTrendChart({ data, macro, label, unit, color, range }) {
  const [hovered, setHovered] = useState(null);
  const values  = data.map(d => d[macro]);
  const max     = Math.max(...values, 1);
  const VW = 280, BARH = 90, LABELH = 18, TOP_PAD = 14;
  const n       = data.length;
  const gap     = n > 15 ? 1 : 2;
  const barW    = (VW - gap * (n - 1)) / n;
  const nonZero = values.filter(v => v > 0);
  const avg     = nonZero.length
    ? Math.round(nonZero.reduce((a, b) => a + b, 0) / nonZero.length)
    : 0;
  const gradId  = `grad-${macro}`;

  return (
    <div style={{ background: "var(--surface)", borderRadius: 16, overflow: "hidden",
      boxShadow: "0 1px 4px rgba(0,0,0,0.06), 0 0 0 1px var(--border)" }}>
      <div style={{ height: 3, background: color }} />
      <div style={{ padding: "14px 18px 16px" }}>
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.6px", color: "var(--muted)" }}>{label}</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 4, marginTop: 2 }}>
            <span style={{ fontSize: 26, fontWeight: 800, color, lineHeight: 1 }}>{avg}</span>
            <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 500 }}>{unit}/day avg</span>
          </div>
        </div>

        <div style={{ position: "relative" }}>
          <svg width="100%" viewBox={`0 0 ${VW} ${TOP_PAD + BARH + LABELH}`}
            style={{ overflow: "visible" }}
            onMouseLeave={() => setHovered(null)}>
            <defs>
              <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   style={{ stopColor: color, stopOpacity: 0.88 }} />
                <stop offset="100%" style={{ stopColor: color, stopOpacity: 0.28 }} />
              </linearGradient>
            </defs>

            {[0.25, 0.5, 0.75].map(pct => {
              const gy = TOP_PAD + BARH - pct * BARH;
              return <line key={pct} x1={0} y1={gy} x2={VW} y2={gy}
                stroke="#C0CBCA" strokeWidth={0.5} strokeDasharray="3,3" />;
            })}
            <line x1={0} y1={TOP_PAD + BARH} x2={VW} y2={TOP_PAD + BARH} stroke="#C0CBCA" strokeWidth={1} />

            {data.map((d, i) => {
              const h          = Math.max(2, (d[macro] / max) * BARH);
              const x          = i * (barW + gap);
              const y          = TOP_PAD + BARH - h;
              const dt         = new Date(d.date + "T12:00:00");
              const isHovered  = hovered === i;
              const showLabel  = range === "weekly" || i === 0 || i === Math.floor(n / 2) || i === n - 1;
              const lbl        = range === "weekly"
                ? ["Su","Mo","Tu","We","Th","Fr","Sa"][dt.getDay()]
                : String(dt.getDate());
              return (
                <g key={d.date} style={{ cursor: d[macro] > 0 ? "pointer" : "default" }}
                  onMouseEnter={() => d[macro] > 0 && setHovered(i)}>
                  <rect x={x} y={y} width={barW} height={h} rx={2}
                    fill={d[macro] > 0 ? `url(#${gradId})` : "#C0CBCA"}
                    fillOpacity={d[macro] > 0 ? (isHovered ? 1 : 0.9) : 0.3}
                    style={{ transition: "fill-opacity 0.1s" }}
                  />
                  {d[macro] > 0 && range === "weekly" && (
                    <text x={x + barW / 2} y={y - 4} textAnchor="middle" fontSize={8}
                      style={{ fill: color }} fontWeight="700">
                      {Math.round(d[macro])}
                    </text>
                  )}
                  {showLabel && (
                    <text x={x + barW / 2} y={TOP_PAD + BARH + 13} textAnchor="middle" fontSize={9}
                      style={{ fill: isHovered ? color : "#888" }} fontWeight={isHovered ? "700" : "400"}>
                      {lbl}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {hovered !== null && (() => {
            const d      = data[hovered];
            const dt     = new Date(d.date + "T12:00:00");
            const dateStr = dt.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
            const pct    = (hovered + 0.5) / n;
            return (
              <div style={{
                position: "absolute", top: 0,
                left: `${pct * 100}%`,
                transform: pct > 0.6 ? "translateX(-100%)" : "translateX(4px)",
                background: "var(--text)", color: "white", borderRadius: 8,
                padding: "5px 10px", fontSize: 11, fontWeight: 600,
                pointerEvents: "none", whiteSpace: "nowrap",
                boxShadow: "0 4px 12px rgba(0,0,0,0.25)", zIndex: 10,
              }}>
                <div style={{ color: "rgba(255,255,255,0.6)", fontSize: 10, marginBottom: 1 }}>{dateStr}</div>
                <div>{Math.round(d[macro])} {unit}</div>
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
}

export default function TrendsTab() {
  const [range,     setRange]     = useState("weekly");
  const [trendData, setTrendData] = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    apiFetch(`/log/trends?range=${range}`)
      .then(d  => setTrendData(d.data))
      .catch(e => setError(e.message || "Failed to load trends"))
      .finally(()  => setLoading(false));
  }, [range]);

  const macros = [
    { key: "calories", label: "Calories", unit: "kcal", color: "var(--orange)", colorLight: "var(--orange-lt)", icon: "local_fire_department" },
    { key: "protein",  label: "Protein",  unit: "g",    color: "var(--teal)",   colorLight: "var(--teal-lt)",   icon: "fitness_center"        },
    { key: "carbs",    label: "Carbs",    unit: "g",    color: "var(--purple)", colorLight: "#EEEBF8",           icon: "grain"                 },
    { key: "fat",      label: "Fat",      unit: "g",    color: "var(--brown)",  colorLight: "var(--brown-lt)",   icon: "water_drop"            },
  ];

  const avgFor = key => {
    if (!trendData) return 0;
    const nz = trendData.map(d => d[key]).filter(v => v > 0);
    return nz.length ? Math.round(nz.reduce((a, b) => a + b, 0) / nz.length) : 0;
  };

  return (
    <div style={{ maxWidth: 960, margin: "0 auto" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "var(--text)" }}>Nutrition Trends</h2>
        <div style={{ display: "flex", gap: 6 }}>
          {["weekly", "monthly"].map(r => (
            <button key={r} onClick={() => setRange(r)}
              style={{ padding: "6px 16px", borderRadius: 20, border: "1.5px solid var(--border)", fontSize: 13, fontWeight: 600,
                cursor: "pointer", transition: "all 0.15s",
                background: range === r ? "var(--teal)" : "var(--surface)",
                color:      range === r ? "white"       : "var(--muted)" }}>
              {r === "weekly" ? "7 Days" : "30 Days"}
            </button>
          ))}
        </div>
      </div>

      {loading && <div style={{ display: "flex", justifyContent: "center", padding: 48 }}><Spin size={28} /></div>}
      {error   && <p style={{ color: "var(--err)", textAlign: "center" }}>{error}</p>}

      {trendData && !loading && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 12, marginBottom: 20 }}>
            {macros.map(m => (
              <div key={m.key} style={{ background: m.colorLight, borderRadius: 14, padding: "12px 14px",
                display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 34, height: 34, borderRadius: 10, background: m.color, flexShrink: 0,
                  display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Icon n={m.icon} size={17} style={{ color: "white" }} />
                </div>
                <div>
                  <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.4px" }}>{m.label}</div>
                  <div style={{ fontSize: 17, fontWeight: 800, color: "var(--text)", lineHeight: 1.15 }}>{avgFor(m.key)}</div>
                  <div style={{ fontSize: 10, color: "var(--muted)" }}>{m.unit}/day</div>
                </div>
              </div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
            {macros.map(m => (
              <MacroTrendChart
                key={m.key}
                data={trendData}
                macro={m.key}
                label={m.label}
                unit={m.unit}
                color={m.color}
                range={range}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
