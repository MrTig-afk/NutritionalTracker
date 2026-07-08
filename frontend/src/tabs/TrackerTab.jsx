import React, { useState, useEffect, useCallback } from "react";
import { apiFetch } from "../lib/api";
import { addDays } from "../lib/nutrition";
import { card, cardHeader, inputStyle, labelStyle, primaryBtn } from "../styles";
import { Icon, Spin } from "../components/Icon";
import MacroBar from "../components/MacroBar";
import DatePicker from "../components/DatePicker";

export default function TrackerTab({ refreshKey, onEditEntry }) {
  const now = new Date();
  const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  const [goals, setGoals] = useState({ calories: 2000, protein: 150, carbs: 250, fat: 65 });
  const [logData, setLogData] = useState(null);
  const [selectedDate, setSelectedDate] = useState(today);
  const [editingGoals, setEditingGoals] = useState(false);
  const [goalDraft, setGoalDraft] = useState({});
  const [loading, setLoading] = useState(true);
  const [savingGoals, setSavingGoals] = useState(false);
  const [deletingId, setDeletingId] = useState(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    try { const [g, l] = await Promise.all([apiFetch("/goals"), apiFetch(`/log?log_date=${selectedDate}`)]); setGoals(g); setLogData(l); }
    catch (e) { console.error("Tracker load failed:", e); }
    finally { setLoading(false); }
  }, [selectedDate]);

  useEffect(() => { loadData(); }, [loadData, refreshKey]);

  const saveGoals = async () => {
    setSavingGoals(true);
    try { const updated = await apiFetch("/goals", { method: "POST", body: JSON.stringify(goalDraft) }); setGoals(updated); setEditingGoals(false); }
    catch (e) { console.error(e); }
    finally { setSavingGoals(false); }
  };

  const deleteEntry = async (logId) => {
    setDeletingId(logId);
    try { await apiFetch(`/log/${logId}`, { method: "DELETE" }); await loadData(); }
    catch (e) { console.error(e); }
    finally { setDeletingId(null); }
  };

  if (loading) return <div style={{ display: "flex", justifyContent: "center", padding: "60px 0" }}><Spin size={24} /></div>;

  const totals = logData?.totals || { calories: 0, protein: 0, carbs: 0, fat: 0 };

  return (
    <div className="ns-tracker-grid">
      {/* Left col — goals + summary */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={card}>
        <div style={cardHeader}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--brown)", display: "flex", alignItems: "center", gap: 8 }}>
            <Icon n="my_location" size={14} style={{ color: "var(--teal)" }} /> Daily Goals
          </div>
          <button onClick={() => { setGoalDraft({ ...goals }); setEditingGoals(v => !v); }}
            style={{ fontSize: 12, fontWeight: 600, color: "var(--teal)", background: "none", border: "none", cursor: "pointer" }}>
            {editingGoals ? "Cancel" : "Edit"}
          </button>
        </div>
        <div style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
          {editingGoals ? (
            <>
              {["calories", "protein", "carbs", "fat", "fibre"].map(key => (
                <div key={key} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <label style={{ ...labelStyle, width: 60, marginBottom: 0 }}>{key}</label>
                  <input type="number" min="0" value={goalDraft[key] || ""} onChange={e => setGoalDraft(prev => ({ ...prev, [key]: parseFloat(e.target.value) || 0 }))} style={{ ...inputStyle, flex: 1 }} />
                  <span style={{ fontSize: 11, color: "var(--muted)", width: 32 }}>{key === "calories" ? "kcal" : "g"}</span>
                </div>
              ))}
              <button onClick={saveGoals} disabled={savingGoals} style={{ ...primaryBtn, opacity: savingGoals ? 0.45 : 1 }}>
                {savingGoals ? "Saving..." : "Save Goals"}
              </button>
            </>
          ) : (
            <>
              <MacroBar label="Calories" current={totals.calories}    goal={goals.calories} color="var(--orange)" />
              <MacroBar label="Protein"  current={totals.protein}     goal={goals.protein}  color="var(--teal)"   />
              <MacroBar label="Carbs"    current={totals.carbs}       goal={goals.carbs}    color="var(--purple)" />
              <MacroBar label="Fat"      current={totals.fat}         goal={goals.fat}      color="var(--brown)"  />
              {goals.fibre > 0 && <MacroBar label="Fibre" current={totals.fibre || 0} goal={goals.fibre} color="var(--mint-dk)" />}
            </>
          )}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        {[
          { label: "Calories", val: totals.calories, unit: "kcal", color: "var(--orange)" },
          { label: "Protein",  val: totals.protein,  unit: "g",    color: "var(--teal)"   },
          { label: "Carbs",    val: totals.carbs,    unit: "g",    color: "var(--purple)"  },
          { label: "Fat",      val: totals.fat,      unit: "g",    color: "var(--brown)"  },
        ].map(({ label, val, unit, color }) => (
          <div key={label} style={{ background: "var(--white)", border: "1px solid var(--border)", borderRadius: 12, padding: "12px 8px", textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.4px" }}>{label}</div>
            <div style={{ fontSize: 20, fontWeight: 700, color, marginTop: 4 }}>{(val || 0).toFixed(0)}</div>
            <div style={{ fontSize: 10, color: "var(--muted)" }}>{unit}</div>
          </div>
        ))}
      </div>
      </div>

      {/* Right col — log entries */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={card}>
        <div style={{ ...cardHeader, background: "var(--off)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <button onClick={() => setSelectedDate(d => addDays(d, -1))} style={{ width: 26, height: 26, borderRadius: 6, border: "1px solid var(--border)", background: "var(--white)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Icon n="chevron_left" size={13} />
            </button>
            <DatePicker value={selectedDate} onChange={setSelectedDate} maxDate={today} />
            <button onClick={() => setSelectedDate(d => addDays(d, 1))} disabled={selectedDate >= today} style={{ width: 26, height: 26, borderRadius: 6, border: "1px solid var(--border)", background: "var(--white)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", opacity: selectedDate >= today ? 0.3 : 1 }}>
              <Icon n="chevron_right" size={13} />
            </button>
          </div>
          <span style={{ fontSize: 11, color: "var(--muted)" }}>{logData?.items?.length || 0} entries</span>
        </div>
        {!logData?.items?.length ? (
          <div style={{ padding: "32px 16px", textAlign: "center", color: "var(--muted)", fontSize: 14, fontStyle: "italic" }}>
            {selectedDate === today ? "No entries yet. Add food from Library." : "No entries for this day."}
          </div>
        ) : (
          logData.items.map((entry, idx) => (
            <div key={entry.log_id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderTop: idx === 0 ? "none" : "1px solid var(--off2)" }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: idx % 2 === 0 ? "var(--teal)" : "var(--orange)", flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{entry.name}</div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
                  ×{entry.servings} serving{entry.servings !== 1 ? "s" : ""} · {entry.contribution.calories.toFixed(0)} kcal · P {entry.contribution.protein.toFixed(1)}g · C {entry.contribution.carbs.toFixed(1)}g · F {entry.contribution.fat.toFixed(1)}g
                </div>
              </div>
              <button onClick={() => onEditEntry && onEditEntry(entry)}
                style={{ width: 30, height: 30, borderRadius: 8, border: "1px solid var(--border)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Icon n="edit" size={13} style={{ color: "var(--teal)" }} />
              </button>
              <button onClick={() => deleteEntry(entry.log_id)} disabled={deletingId === entry.log_id}
                style={{ width: 30, height: 30, borderRadius: 8, border: "1px solid var(--border)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
                {deletingId === entry.log_id ? <Spin size={13} color="var(--muted)" /> : <Icon n="delete" size={13} style={{ color: "var(--danger)" }} />}
              </button>
            </div>
          ))
        )}
      </div>
      </div>
    </div>
  );
}
