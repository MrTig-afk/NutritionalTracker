import React, { useState, useEffect } from "react";
import { apiFetch } from "../lib/api";
import { Icon, Spin } from "../components/Icon";

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
    { key: "calories", label: "Cal",     unit: "kcal", color: "var(--orange)", icon: "local_fire_department" },
    { key: "protein",  label: "Protein", unit: "g",    color: "var(--teal)",   icon: "fitness_center"        },
    { key: "carbs",    label: "Carbs",   unit: "g",    color: "var(--purple)", icon: "grain"                 },
    { key: "fat",      label: "Fat",     unit: "g",    color: "var(--brown)",  icon: "water_drop"            },
  ];

  const avgFor = key => {
    if (!trendData) return 0;
    const nz = trendData.map(d => d[key]).filter(v => v > 0);
    return nz.length ? Math.round(nz.reduce((a, b) => a + b, 0) / nz.length) : 0;
  };

  const today = new Date().toISOString().split("T")[0];

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
      {error   && <p style={{ color: "var(--danger)", textAlign: "center" }}>{error}</p>}

      {trendData && !loading && (
        <>
          {/* Average summary cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 20 }}>
            {macros.map(m => (
              <div key={m.key} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 14, padding: "12px 8px", textAlign: "center" }}>
                <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.4px" }}>{m.label}</div>
                <div style={{ fontSize: 20, fontWeight: 800, color: m.color, marginTop: 4 }}>{avgFor(m.key)}</div>
                <div style={{ fontSize: 10, color: "var(--muted)" }}>{m.unit}/day avg</div>
              </div>
            ))}
          </div>

          {/* Daily breakdown table */}
          <div style={{ background: "var(--surface)", borderRadius: 14, border: "1px solid var(--border)", overflow: "hidden" }}>
            {/* Column headers */}
            <div style={{ display: "grid", gridTemplateColumns: "72px repeat(4, 1fr)", padding: "8px 14px", background: "var(--off)", borderBottom: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 700, textTransform: "uppercase" }}>Date</div>
              {macros.map(m => (
                <div key={m.key} style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", color: m.color, textAlign: "center" }}>{m.label}</div>
              ))}
            </div>

            {[...trendData].reverse().map((day, i, arr) => {
              const dt = new Date(day.date + "T12:00:00");
              const isToday = day.date === today;
              const hasData = macros.some(m => day[m.key] > 0);
              return (
                <div key={day.date} style={{
                  display: "grid", gridTemplateColumns: "72px repeat(4, 1fr)",
                  padding: "10px 14px",
                  borderBottom: i < arr.length - 1 ? "1px solid var(--border)" : "none",
                  background: isToday ? "var(--teal-lt)" : "transparent",
                  alignItems: "center",
                }}>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: isToday ? "var(--teal)" : "var(--text)" }}>
                      {dt.toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                    </div>
                    <div style={{ fontSize: 10, color: "var(--muted)" }}>
                      {dt.toLocaleDateString("en-US", { weekday: "short" })}
                    </div>
                  </div>
                  {macros.map(m => (
                    <div key={m.key} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color: hasData && day[m.key] > 0 ? m.color : "var(--border)" }}>
                        {day[m.key] > 0 ? Math.round(day[m.key]) : "—"}
                      </div>
                      {day[m.key] > 0 && <div style={{ fontSize: 9, color: "var(--muted)" }}>{m.unit}</div>}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
