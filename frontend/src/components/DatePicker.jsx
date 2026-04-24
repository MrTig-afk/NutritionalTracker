import React, { useState, useRef, useEffect } from "react";
import { apiFetch } from "../lib/api";
import { formatDisplayDate } from "../lib/nutrition";
import { Icon } from "./Icon";

const MONTHS_LONG = ["January","February","March","April","May","June","July","August","September","October","November","December"];
const DAYS_SHORT  = ["Su","Mo","Tu","We","Th","Fr","Sa"];

export default function DatePicker({ value, onChange, maxDate }) {
  const [open,         setOpen]         = useState(false);
  const [dropPos,      setDropPos]      = useState(null);
  const [viewYear,     setViewYear]     = useState(() => Number(value.split("-")[0]));
  const [viewMonth,    setViewMonth]    = useState(() => Number(value.split("-")[1]) - 1);
  const [trackedDates, setTrackedDates] = useState(new Set());
  const btnRef  = useRef(null);
  const dropRef = useRef(null);

  const handleToggle = () => {
    if (!open && btnRef.current) {
      const r        = btnRef.current.getBoundingClientRect();
      const vw       = window.innerWidth;
      const vh       = window.innerHeight;
      const margin   = 8;
      const calW     = Math.min(272, vw - margin * 2);
      const calHEst  = 320;
      const bottomNav = 72;

      const idealLeft = r.left + r.width / 2 - calW / 2;
      const left = Math.max(margin, Math.min(idealLeft, vw - calW - margin));

      const spaceBelow = vh - r.bottom - margin - bottomNav;
      const top = spaceBelow >= calHEst
        ? r.bottom + margin
        : Math.max(margin, r.top - margin - calHEst);

      setDropPos({ top, left, calW });
    }
    setOpen(o => !o);
  };

  useEffect(() => {
    if (!open) return;
    apiFetch(`/log/calendar?year=${viewYear}&month=${viewMonth + 1}`)
      .then(d => setTrackedDates(new Set(d.dates)))
      .catch(() => {});
  }, [open, viewYear, viewMonth]);

  useEffect(() => {
    if (!open) return;
    const close = e => {
      if (!btnRef.current?.contains(e.target) && !dropRef.current?.contains(e.target))
        setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("touchstart", close, { passive: true });
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("touchstart", close);
    };
  }, [open]);

  const goPrev = () => {
    if (viewMonth === 0) { setViewMonth(11); setViewYear(y => y - 1); }
    else setViewMonth(m => m - 1);
  };
  const goNext = () => {
    if (viewMonth === 11) { setViewMonth(0); setViewYear(y => y + 1); }
    else setViewMonth(m => m + 1);
  };
  const nextDisabled = (() => {
    const [my, mm] = maxDate.split("-").map(Number);
    return viewYear > my || (viewYear === my && viewMonth >= mm - 1);
  })();

  const daysInMonth  = new Date(viewYear, viewMonth + 1, 0).getDate();
  const startWeekday = new Date(viewYear, viewMonth, 1).getDay();
  const cells = [...Array(startWeekday).fill(null),
                 ...Array.from({ length: daysInMonth }, (_, i) => i + 1)];

  return (
    <div style={{ display: "inline-block" }}>
      <button ref={btnRef} onClick={handleToggle}
        style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 700,
          color: "var(--brown)", background: "none", border: "none", cursor: "pointer",
          padding: "3px 6px", borderRadius: 8,
          outline: open ? "2px solid var(--teal)" : "none", outlineOffset: 1 }}>
        <Icon n="calendar_today" size={14} style={{ color: "var(--teal)" }} />
        {formatDisplayDate(value)}
        <Icon n="expand_more" size={13} style={{ color: "var(--muted)",
          transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s" }} />
      </button>

      {open && dropPos && (
        <>
          <div onClick={() => setOpen(false)} style={{
            position: "fixed", inset: 0, zIndex: 999,
            background: "rgba(0,0,0,0.18)",
            display: window.innerWidth < 600 ? "block" : "none",
          }} />
          <div ref={dropRef} className="calendar-drop" style={{
            position: "fixed", top: dropPos.top, left: dropPos.left, zIndex: 1000,
            background: "var(--surface)", border: "1px solid var(--border)",
            borderRadius: 16, boxShadow: "0 8px 32px rgba(0,0,0,0.16)",
            padding: "14px 16px", width: dropPos.calW,
          }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            <button onClick={goPrev}
              style={{ width: 28, height: 28, borderRadius: 8, border: "1px solid var(--border)", background: "none", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Icon n="chevron_left" size={14} />
            </button>
            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)" }}>
              {MONTHS_LONG[viewMonth]} {viewYear}
            </span>
            <button onClick={goNext} disabled={nextDisabled}
              style={{ width: 28, height: 28, borderRadius: 8, border: "1px solid var(--border)", background: "none", cursor: nextDisabled ? "default" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", opacity: nextDisabled ? 0.3 : 1 }}>
              <Icon n="chevron_right" size={14} />
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", marginBottom: 4 }}>
            {DAYS_SHORT.map(d => (
              <div key={d} style={{ textAlign: "center", fontSize: 10, fontWeight: 700, color: "var(--muted)", padding: "2px 0" }}>{d}</div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 2 }}>
            {cells.map((day, idx) => {
              if (!day) return <div key={`_${idx}`} />;
              const iso        = `${viewYear}-${String(viewMonth + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
              const isSelected = iso === value;
              const isToday    = iso === maxDate;
              const isFuture   = iso > maxDate;
              const isTracked  = trackedDates.has(iso);
              return (
                <button key={iso} disabled={isFuture}
                  onClick={() => { onChange(iso); setOpen(false); }}
                  style={{
                    width: "100%", border: "none", borderRadius: 8,
                    padding: "7px 0", minHeight: 36,
                    display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
                    fontSize: 12, fontWeight: isSelected || isToday ? 700 : 400,
                    cursor: isFuture ? "default" : "pointer",
                    background:    isSelected ? "var(--teal)" : "transparent",
                    color:         isSelected ? "white" : isFuture ? "var(--border)" : isToday ? "var(--teal)" : "var(--text)",
                    outline:       isToday && !isSelected ? "1.5px solid var(--teal)" : "none",
                    outlineOffset: -1,
                  }}>
                  <span>{day}</span>
                  <div style={{
                    width: 4, height: 4, borderRadius: "50%",
                    background: isTracked
                      ? (isSelected ? "rgba(255,255,255,0.85)" : "var(--teal)")
                      : "transparent",
                  }} />
                </button>
              );
            })}
          </div>

          {value !== maxDate && (
            <button onClick={() => { onChange(maxDate); setOpen(false); }}
              style={{ width: "100%", marginTop: 10, padding: "7px", background: "var(--teal-lt)", border: "1px solid var(--teal)", borderRadius: 10, fontSize: 12, fontWeight: 700, color: "var(--teal)", cursor: "pointer" }}>
              Jump to Today
            </button>
          )}
        </div>
        </>
      )}
    </div>
  );
}
