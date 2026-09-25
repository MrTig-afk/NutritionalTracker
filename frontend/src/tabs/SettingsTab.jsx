import React, { useState, useEffect } from "react";
import { apiFetch, downloadExport } from "../lib/api";
import { card, inputStyle, ghostBtn, pillRow, errorBanner } from "../styles";
import { Icon, Spin, Spark } from "../components/Icon";
import { pushSupported, getPermission, getSubscribed, enablePush, disablePush } from "../lib/push";
import { useEnergyUnit, deletionDate, todayLocal } from "../lib/nutrition";
import ConnectedAppsCard from "../components/ConnectedAppsCard";

const LINKS = {
  email:    "mailto:kaushiknaru2002@gmail.com",
  linkedin: "https://www.linkedin.com/in/kaushikn2002/",
  github:   "https://github.com/MrTig-afk",
};

const NOTIF_TYPES = [
  { key: "meal_morning",   label: "Morning reminder",   desc: "Log your first meal",     hasTime: true },
  { key: "meal_afternoon", label: "Afternoon reminder", desc: "Log your second meal",    hasTime: true },
  { key: "meal_evening",   label: "Evening reminder",   desc: "Finish logging your day", hasTime: true },
  { key: "weekly_summary", label: "Weekly summary",     desc: "Sunday evening recap of your week" },
];

// 12-hour display ("9:00 AM") <-> canonical 24h storage ("09:00")
const to12h = (hhmm) => {
  const [h, m] = String(hhmm).split(":").map(Number);
  if (Number.isNaN(h)) return "";
  const ap = h >= 12 ? "PM" : "AM";
  return `${h % 12 || 12}:${String(m || 0).padStart(2, "0")} ${ap}`;
};
const parse12h = (text) => {
  const m = String(text).trim().toUpperCase().match(/^(\d{1,2})(?:[:.](\d{2}))?\s*(AM|PM)$/);
  if (!m) return null;
  let h = parseInt(m[1], 10);
  const min = m[2] ? parseInt(m[2], 10) : 0;
  if (h < 1 || h > 12 || min > 59) return null;
  if (m[3] === "PM" && h !== 12) h += 12;
  if (m[3] === "AM" && h === 12) h = 0;
  return `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")}`;
};

const LinkedInLogo = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
    <path fill="#0A66C2" d="M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45z"/>
  </svg>
);

const GitHubLogo = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
    <path fill="currentColor" d="M12 .5A11.5 11.5 0 0 0 .5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56v-2.17c-3.2.7-3.87-1.37-3.87-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.69 1.25 3.35.96.1-.75.4-1.25.72-1.54-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.78 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.24 2.76.12 3.05.74.8 1.18 1.83 1.18 3.09 0 4.43-2.69 5.4-5.26 5.68.41.36.78 1.05.78 2.13v3.16c0 .31.2.68.8.56A11.5 11.5 0 0 0 23.5 12 11.5 11.5 0 0 0 12 .5z"/>
  </svg>
);

function Toggle({ on, onChange, disabled, label, write = true }) {
  return (
    <button
      data-write={write ? "" : undefined}
      onClick={onChange}
      disabled={disabled}
      aria-pressed={on}
      aria-label={label}
      style={{
        width: 44, height: 26, borderRadius: 20, border: "none", padding: 2,
        background: on ? "var(--teal)" : "var(--border)",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.45 : 1,
        display: "flex", justifyContent: on ? "flex-end" : "flex-start",
        transition: "background 0.15s", flexShrink: 0,
      }}>
      <div style={{ width: 22, height: 22, borderRadius: "50%", background: "var(--white)", boxShadow: "0 1px 3px rgba(0,0,0,0.25)" }} />
    </button>
  );
}

// ConnectedAppsCard's own marker ("connectedAppsSeen"): only a device that has shown the owner's card may see its load error
const seenAsOwner = () => { try { return !!localStorage.getItem("connectedAppsSeen"); } catch { return false; } };

// Artifact v10 lane S: a list of rows under group headings (S1); a row opens its own screen with "‹ Settings" (S2, S3).
const groupHead = { fontSize: 11, fontWeight: 800, letterSpacing: 0.6, color: "var(--muted)", textTransform: "uppercase", margin: "4px 4px -8px" };

function Row({ icon, name, value, onClick, write, danger, control, first }) {
  const body = (
    <>
      <span style={{ width: 32, height: 32, borderRadius: 9, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center",
        background: danger ? "var(--danger-lt)" : "var(--off)", border: danger ? "none" : "1px solid var(--border)" }}>{icon}</span>
      <span style={{ flex: 1, minWidth: 0, textAlign: "left" }}>
        <span style={{ display: "block", fontSize: 14, fontWeight: 700, color: danger ? "var(--danger)" : "var(--text)" }}>{name}</span>
        {value && <span style={{ display: "block", fontSize: 12, color: "var(--muted)", marginTop: 2 }}>{value}</span>}
      </span>
      {control ?? <Icon n="chevron_right" size={18} style={{ color: "var(--muted)", flexShrink: 0 }} />}
    </>
  );
  const style = { display: "flex", alignItems: "center", gap: 12, width: "100%", padding: "12px 16px", background: "none", border: "none",
    borderTop: first ? "none" : "1px solid var(--border)", color: "inherit", font: "inherit" };
  return control
    ? <div style={style}>{body}</div>
    : <button data-write={write ? "" : undefined} onClick={onClick} style={{ ...style, cursor: "pointer" }}>{body}</button>;
}

export default function SettingsTab({ setEnergyUnit, onDeleted, pending, startAt, onStarted }) {
  const [screen, setScreen] = useState(null);   // null = the list (S1)
  const [appCount, setAppCount] = useState(null);   // number, "error", or false for "not this account"

  // A screen is one history step, so the phone's back gesture returns to the list (S2). Only ever one level deep:
  // opening a screen reuses our step (an open screen's, or the one a tab switch left behind) instead of stacking.
  const openScreen = (k) => {
    const ours = window.history.state?.nsSettings || window.history.state?.nsSettingsGone;
    window.history[ours ? "replaceState" : "pushState"]({ nsSettings: k }, "");
    setScreen(k);
  };
  const closeScreen = () => { if (window.history.state?.nsSettings) window.history.back(); else setScreen(null); };
  useEffect(() => {
    const onPop = () => setScreen(window.history.state?.nsSettings || null);
    window.addEventListener("popstate", onPop);
    return () => {
      window.removeEventListener("popstate", onPop);
      // leaving the tab with a screen open: the step must not reopen that screen later
      if (window.history.state?.nsSettings) window.history.replaceState({ nsSettingsGone: true }, "");
    };
  }, []);
  useEffect(() => {   // the deletion banner's Export button (L2) opens S3 directly
    if (!startAt) return;
    openScreen(startAt);
    onStarted();
  }, [startAt]);   // eslint-disable-line react-hooks/exhaustive-deps

  const onList = screen === null;
  useEffect(() => {
    if (!onList) return;
    apiFetch("/settings/connected-apps").then(a => setAppCount(a.length))
      // 403: connecting is the owner's only (H4b); any other error shows only where this device has seen the card
      .catch(e => setAppCount(e.status !== 403 && seenAsOwner() ? "error" : false));
  }, [onList]);

  const energyUnit = useEnergyUnit();
  const [unitErr, setUnitErr]         = useState(false);
  const [unitSaving, setUnitSaving]   = useState(false);
  const [permission, setPermission]   = useState(getPermission());
  const [subscribed, setSubscribed]   = useState(false);
  const [pushLoading, setPushLoading] = useState(false);
  const [prefs, setPrefs]             = useState(null);
  const [deleting, setDeleting]       = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [error, setError]             = useState(null);
  const [timeDrafts, setTimeDrafts]   = useState({});
  const [timeErrors, setTimeErrors]   = useState({});
  const [health, setHealth]           = useState(null);
  const [alertState, setAlertState]   = useState(null); // null | "sending" | "sent" | "failed"
  const [askFirst, setAskFirst]       = useState(null);  // G4: null while loading
  const [askErr, setAskErr]           = useState(false);
  const [exporting, setExporting]     = useState(null);  // G3: "xlsx" | "csv" while the file is made
  const [exportErr, setExportErr]     = useState(null);

  const [askLoadErr, setAskLoadErr]   = useState(false);
  useEffect(() => { apiFetch("/settings/library").then(r => setAskFirst(r.ask_before_saving)).catch(() => setAskLoadErr(true)); }, []);

  const toggleAskFirst = () => {
    const next = !askFirst;
    setAskFirst(next); setAskErr(false);
    apiFetch("/settings/library", { method: "PUT", body: JSON.stringify({ ask_before_saving: next }) })
      .catch(() => { setAskFirst(!next); setAskErr(true); });
  };

  const exportAs = async (format) => {
    setExporting(format); setExportErr(null);
    try { await downloadExport(format); }
    catch (e) { setExportErr(e.message); }
    finally { setExporting(null); }
  };

  useEffect(() => { apiFetch("/settings/admin/health").then(setHealth).catch(() => {}); }, []);

  const chooseUnit = (u) => {
    if (u === energyUnit || unitSaving) return;   // one save at a time, so the server keeps the last tap
    setEnergyUnit(u); setUnitErr(false); setUnitSaving(true);
    apiFetch("/settings/energy-unit", { method: "PUT", body: JSON.stringify({ unit: u }) })
      .catch(() => {
        setUnitErr(true);
        apiFetch("/settings/energy-unit").then(r => setEnergyUnit(r.unit === "kJ" ? "kJ" : "kcal")).catch(() => setEnergyUnit(energyUnit));
      })
      .finally(() => setUnitSaving(false));
  };

  const sendTestAlert = async () => {
    setAlertState("sending");
    try { await apiFetch("/settings/admin/test-alert", { method: "POST" }); setAlertState("sent"); }
    catch { setAlertState("failed"); }
  };

  useEffect(() => {
    getSubscribed().then(setSubscribed).catch(() => {});
    apiFetch("/settings/notifications").then(r => {
      setPrefs(r.prefs);
      const drafts = {};
      NOTIF_TYPES.filter(t => t.hasTime).forEach(t => { drafts[t.key] = to12h(r.prefs[`${t.key}_time`]); });
      setTimeDrafts(drafts);
    }).catch(() => setPrefs({}));
  }, []);

  const toggleMaster = async () => {
    setPushLoading(true); setError(null);
    try {
      if (subscribed) {
        await disablePush();
        setSubscribed(false);
      } else {
        const perm = await enablePush();
        setPermission(perm);
        if (perm === "granted") setSubscribed(true);
      }
    } catch (e) { setError(e.message); }
    finally { setPushLoading(false); }
  };

  const togglePref = (key) => {
    const next = { ...prefs, [key]: !prefs[key] };
    setPrefs(next);
    apiFetch("/settings/notifications", { method: "PUT", body: JSON.stringify({ prefs: next }) })
      .catch(() => setPrefs(prefs)); // revert on failure
  };

  const commitTime = (key) => {
    const timeKey   = `${key}_time`;
    const canonical = parse12h(timeDrafts[key]);
    if (!canonical) {
      setTimeErrors(e => ({ ...e, [key]: true }));
      setTimeDrafts(d => ({ ...d, [key]: to12h(prefs[timeKey]) })); // revert
      return;
    }
    setTimeErrors(e => ({ ...e, [key]: false }));
    setTimeDrafts(d => ({ ...d, [key]: to12h(canonical) }));
    if (canonical === prefs[timeKey]) return;
    const next = { ...prefs, [timeKey]: canonical };
    setPrefs(next);
    apiFetch("/settings/notifications", { method: "PUT", body: JSON.stringify({ prefs: next }) })
      .catch(() => { setPrefs(prefs); setTimeDrafts(d => ({ ...d, [key]: to12h(prefs[timeKey]) })); });
  };

  const deleteAccount = async () => {
    if (confirmText !== "DELETE") return;
    setDeleting(true); setError(null);
    try {
      const r = await apiFetch("/account", { method: "DELETE" });
      setConfirmOpen(false);
      onDeleted(r.delete_after);   // L2: this device stays signed in, view-only, with the banner
    } catch (e) {
      setConfirmOpen(false);   // the error banner sits behind the modal otherwise
      setError(e.message);
    } finally {
      setDeleting(false);
    }
  };

  const rowStyle = { display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderTop: "1px solid var(--border)" };

  const unitBody = (
    <div style={{ ...card, padding: "12px 16px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 11, color: "var(--muted)" }}>How calories are shown. Stored the same either way.</div>
      <div style={{ ...pillRow, background: "var(--off)" }}>
        {["kcal", "kJ"].map(u => (
          <button data-write key={u} onClick={() => chooseUnit(u)} aria-pressed={energyUnit === u} disabled={unitSaving}
            style={energyUnit === u
              ? { flex: 1, padding: "7px", background: "var(--teal)", color: "white", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: "pointer" }
              : { flex: 1, padding: "7px", background: "transparent", color: "var(--muted)", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
            {u}
          </button>
        ))}
      </div>
      {unitErr && <div style={{ fontSize: 11, color: "var(--danger)", marginTop: 6 }}>Couldn't save. Try again.</div>}
    </div>
  );

  const exportBody = (
    <div style={{ ...card, padding: "12px 16px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 12, color: "var(--muted)" }}>Download everything: food log, goals, meal templates and your Library.</div>
      <div style={{ display: "flex", gap: 8 }}>
        {[["xlsx", "Excel (.xlsx)"], ["csv", "CSV (.zip)"]].map(([f, label], i) => (
          <button key={f} onClick={() => exportAs(f)} disabled={!!exporting}
            style={{ flex: 1, padding: "10px", borderRadius: 10, fontSize: 13, fontWeight: 700, cursor: exporting ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
              ...(i === 0 ? { background: "var(--teal)", color: "#fff", border: "none" }
                          : { background: "transparent", color: "var(--text2)", border: "1px solid var(--border)" }) }}>
            {exporting === f ? <Spin size={14} color={i === 0 ? "white" : undefined} /> : label}
          </button>
        ))}
      </div>
      <div style={{ fontSize: 11, color: "var(--muted)" }}>nutriscan-export-{todayLocal()}.xlsx</div>
      {exportErr && <div style={errorBanner}>{exportErr}</div>}
    </div>
  );

  const contactBody = (
    <div style={{ ...card, display: "flex", gap: 14, padding: "16px", justifyContent: "center" }}>
      {[
        { href: LINKS.email,    title: "Email",    inner: <Icon n="mail" size={20} style={{ color: "var(--accent)" }} /> },
        { href: LINKS.linkedin, title: "LinkedIn", inner: <LinkedInLogo /> },
        { href: LINKS.github,   title: "GitHub",   inner: <GitHubLogo /> },
      ].map(l => (
        <a key={l.title} href={l.href} target="_blank" rel="noopener noreferrer" title={l.title}
          style={{ width: 48, height: 48, borderRadius: "50%", background: "var(--off)", border: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text)" }}>
          {l.inner}
        </a>
      ))}
    </div>
  );

  const notifyBody = (
    <div style={card}>
      <div style={{ ...rowStyle, borderTop: "none" }}>
        <Icon n={subscribed ? "notifications_active" : "notifications"} size={20} style={{ color: "var(--accent)", flexShrink: 0 }} />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 700 }}>Notifications on this device</div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
            {permission === "denied"
              ? "Blocked in your browser/system settings"
              : subscribed ? "Enabled" : "Turn on to receive any notifications"}
          </div>
        </div>
        {pushLoading ? <Spin size={18} /> : (
          <Toggle label="Push notifications" on={subscribed} onChange={toggleMaster} disabled={permission === "denied" || !pushSupported()}
            write={!subscribed} /* turning notifications off still works inside the 15 days */ />
        )}
      </div>

      <div style={{ padding: "8px 16px", fontSize: 11, color: "var(--muted)", borderTop: "1px solid var(--border)", lineHeight: 1.5 }}>
        Goal-reached and scan-limit alerts are always included. Reminders below are optional — turn on the ones you want.
      </div>

      {NOTIF_TYPES.map(t => (
        <div key={t.key} style={{ ...rowStyle, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 150 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: subscribed ? "var(--text)" : "var(--muted)" }}>{t.label}</div>
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
              {t.hasTime && prefs?.[t.key] ? `Reminds you at ${to12h(prefs[`${t.key}_time`])} — ${t.desc.toLowerCase()}` : t.desc}
            </div>
          </div>
          {prefs === null ? <Spin size={14} /> : (
            <Toggle label={t.label} on={!!prefs[t.key]} onChange={() => togglePref(t.key)} disabled={!subscribed} />
          )}
          {t.hasTime && prefs?.[t.key] && subscribed && (
            <div style={{ flexBasis: "100%", display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>Time:</span>
              <input
                data-write
                value={timeDrafts[t.key] ?? ""}
                onChange={e => setTimeDrafts(d => ({ ...d, [t.key]: e.target.value }))}
                onBlur={() => commitTime(t.key)}
                onKeyDown={e => e.key === "Enter" && e.target.blur()}
                placeholder="9:00 AM"
                style={{ width: 110, padding: "6px 10px", fontSize: 16, borderRadius: 8, background: "var(--off)", color: "var(--text)", border: `1.5px solid ${timeErrors[t.key] ? "var(--danger)" : "var(--border)"}` }}
              />
              {timeErrors[t.key] && (
                <span style={{ fontSize: 11, color: "var(--danger)" }}>Use a time like 9:00 AM</span>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );

  // Admin (owner only): renders nothing until GET /settings/admin/health returns 200
  const adminBody = health && (
    <div style={{ ...card, padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div style={{ background: "var(--off)", borderRadius: 10, padding: "8px 10px" }}>
          <div style={{ fontSize: 11, color: "var(--muted)" }}>API requests left today</div>
          {/* 200 is api_v1.DAILY_CAP */}
          <div style={{ fontSize: 16, fontWeight: 800, color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>{health.requests_left_today} / 200</div>
          <div style={{ fontSize: 11, color: "var(--muted)" }}>resets {new Date(health.resets_at).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}</div>
        </div>
        <div style={{ background: "var(--off)", borderRadius: 10, padding: "8px 10px" }}>
          <div style={{ fontSize: 11, color: "var(--muted)" }}>Neon budget used</div>
          <div style={{ fontSize: 16, fontWeight: 800, color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>{health.neon_budget_percent}%</div>
          <div style={{ fontSize: 11, color: "var(--muted)" }}>resets {new Date(health.budget_period_resets + "T00:00:00").toLocaleDateString("en-AU", { day: "numeric", month: "short" })}</div>
        </div>
      </div>
      {health.api_paused
        ? <div style={{ fontSize: 12, fontWeight: 700, borderRadius: 10, padding: "8px 12px", background: "var(--orange-lt)", color: "var(--orange)" }}>API paused: Neon budget at 90%</div>
        : <div style={{ fontSize: 12, fontWeight: 700, borderRadius: 10, padding: "8px 12px", background: "var(--off)", color: "var(--mint-dk)" }}>API running</div>}
      <div style={{ fontSize: 11, color: "var(--muted)" }}>At 90% of the Neon budget the Claude API pauses itself; the app keeps working.</div>
      <button data-write onClick={sendTestAlert} disabled={alertState === "sending"} style={{ ...ghostBtn, opacity: alertState === "sending" ? 0.6 : 1 }}>
        {alertState === "sending" ? <Spin size={14} /> : <Icon n="notifications_active" size={14} />}
        Send test alert
      </button>
      {alertState === "sent" && <div style={{ fontSize: 12, fontWeight: 700, color: "var(--mint-dk)" }}>Sent. It should arrive on every device with notifications on.</div>}
      {alertState === "failed" && <div style={errorBanner}>Couldn't send the test alert.</div>}
    </div>
  );

  const SCREENS = {
    unit:    { title: "Energy unit", body: unitBody },
    notify:  { title: "Notifications", body: notifyBody },
    apps:    { title: null, body: <ConnectedAppsCard /> },   // the card carries its own "Connected apps" header
    export:  { title: "Export my data", body: exportBody },
    contact: { title: "Contact the developer", body: contactBody },
    admin:   { title: "Admin", body: adminBody },
  };

  const reminders = prefs ? NOTIF_TYPES.filter(t => prefs[t.key]).length : 0;
  const notifyValue = permission === "denied" ? "Blocked in your browser settings"
    : subscribed ? `On · ${reminders} reminder${reminders === 1 ? "" : "s"}` : "Off on this device";
  const appsValue = appCount === "error" ? "Couldn't load" : appCount ? `${appCount} connected` : "Nothing connected yet";
  const ico = (n) => <Icon n={n} size={17} style={{ color: "var(--accent)" }} />;

  const open = SCREENS[screen];
  return (
    <div style={{ maxWidth: 560, margin: "0 auto", padding: "0 16px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
      {open ? (
        <>
          <button onClick={closeScreen} style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: 2, background: "none", border: "none",
            color: "var(--accent)", fontSize: 14, fontWeight: 700, cursor: "pointer", padding: 0 }}>
            <Icon n="chevron_left" size={20} /> Settings
          </button>
          {open.title && <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text)", marginTop: -6 }}>{open.title}</div>}
          {open.body}
        </>
      ) : (
        <>
          <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text)" }}>Settings</div>

          <div style={groupHead}>Display</div>
          <div style={card}><Row first write icon={ico("straighten")} name="Energy unit" value={energyUnit} onClick={() => openScreen("unit")} /></div>

          <div style={groupHead}>Library</div>
          <div style={card}>
            <Row first icon={ico("bookmark_add")} name="Ask before saving new foods"
              value={askFirst === null ? (askLoadErr ? "Couldn't load" : "") : `${askFirst ? "On" : "Off"}${health ? " · the app and Claude share one Library" : ""}`}
              control={askFirst === null ? (askLoadErr ? <span /> : <Spin size={14} />) : <Toggle label="Ask before saving new foods to my Library" on={askFirst} onChange={toggleAskFirst} />} />
            {askErr && <div style={{ fontSize: 11, color: "var(--danger)", padding: "0 16px 12px" }}>Couldn't save. Try again.</div>}
          </div>

          <div style={groupHead}>Reminders</div>
          <div style={card}><Row first icon={ico(subscribed ? "notifications_active" : "notifications")} name="Notifications" value={notifyValue} onClick={() => openScreen("notify")} /></div>

          {appCount !== null && appCount !== false && <>
            <div style={groupHead}>Claude</div>
            <div style={card}><Row first icon={<Spark size={16} color="var(--claude)" />} name="Connected apps" value={appsValue} onClick={() => openScreen("apps")} /></div>
          </>}

          <div style={groupHead}>Your data</div>
          <div style={card}>
            <Row first icon={ico("download")} name="Export my data" value="Excel or CSV" onClick={() => openScreen("export")} />
            <Row icon={ico("mail")} name="Contact the developer" value="Email, LinkedIn, GitHub" onClick={() => openScreen("contact")} />
          </div>

          {health && <>
            <div style={groupHead}>Admin · only you</div>
            <div style={card}><Row first icon={ico("admin_panel_settings")} name="Admin"
              value={`${health.api_paused ? "API paused" : "API running"} · Neon ${health.neon_budget_percent}%`} onClick={() => openScreen("admin")} /></div>
          </>}

          {!pending && (
            <div style={{ ...card, border: "1px solid var(--danger)" }}>
              <Row first write danger icon={<Icon n="delete" size={17} style={{ color: "var(--danger)" }} />} name="Delete my account"
                onClick={() => { setConfirmOpen(true); setConfirmText(""); }} />
            </div>
          )}
        </>
      )}

      {error && (
        <div style={{ background: "var(--danger-lt)", border: "1px solid var(--danger)", borderRadius: 10, padding: "10px 14px", fontSize: 13, color: "var(--danger)" }}>
          {error}
        </div>
      )}

      {/* Delete confirmation modal */}
      {confirmOpen && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)", zIndex: 80, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
          <div style={{ background: "var(--surface)", borderRadius: 20, padding: 24, width: "100%", maxWidth: 400 }}>
            <div style={{ fontSize: 17, fontWeight: 800, color: "var(--danger)" }}>Delete your account?</div>
            <div style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.6, margin: "10px 0 6px" }}>
              Your account and everything in it will be deleted on{" "}
              <strong style={{ color: "var(--text)" }}>{deletionDate(Date.now() + 15 * 86400000)}</strong>.
              Until then you can still see and export everything, and sign in to keep it.
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>Type DELETE to confirm</div>
            <input
              value={confirmText}
              onChange={e => setConfirmText(e.target.value)}
              placeholder="DELETE"
              autoFocus
              style={{ ...inputStyle, fontSize: 16 }}
            />
            <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
              <button onClick={() => setConfirmOpen(false)} disabled={deleting}
                style={{ flex: 1, padding: "12px", background: "var(--off)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 13, fontWeight: 700, cursor: "pointer" }}>
                Cancel
              </button>
              <button onClick={deleteAccount} disabled={confirmText !== "DELETE" || deleting}
                style={{ flex: 1, padding: "12px", background: "var(--danger)", color: "var(--on-danger)", border: "none", borderRadius: 10, fontSize: 13, fontWeight: 700, cursor: confirmText === "DELETE" ? "pointer" : "not-allowed", opacity: confirmText === "DELETE" ? 1 : 0.5, display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
                {deleting ? <Spin size={14} color="white" /> : "Delete in 15 days"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
