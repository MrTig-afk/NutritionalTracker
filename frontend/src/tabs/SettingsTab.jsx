import React, { useState, useEffect } from "react";
import { apiFetch, supabase } from "../lib/api";
import { card, cardHeader, inputStyle } from "../styles";
import { Icon, Spin } from "../components/Icon";
import { pushSupported, getPermission, getSubscribed, enablePush, disablePush } from "../lib/push";

const LINKS = {
  email:    "mailto:theimpracticalguy007@gmail.com",
  linkedin: "https://www.linkedin.com/in/kaushikn2002/",
  github:   "https://github.com/MrTig-afk",
};

const NOTIF_TYPES = [
  { key: "meal_morning",   label: "Morning reminder",   desc: "9am — log your first meal" },
  { key: "meal_afternoon", label: "Afternoon reminder", desc: "2pm — log your second meal" },
  { key: "meal_evening",   label: "Evening reminder",   desc: "9pm — finish logging your day" },
  { key: "goal_reached",   label: "Goal reached",       desc: "When you hit your daily calorie goal" },
  { key: "scan_limit",     label: "Scan limit",         desc: "When you've used all daily scans" },
  { key: "weekly_summary", label: "Weekly summary",     desc: "Sunday evening recap of your week" },
];

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

function Toggle({ on, onChange, disabled }) {
  return (
    <button
      onClick={onChange}
      disabled={disabled}
      aria-pressed={on}
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

export default function SettingsTab() {
  const [permission, setPermission]   = useState(getPermission());
  const [subscribed, setSubscribed]   = useState(false);
  const [pushLoading, setPushLoading] = useState(false);
  const [prefs, setPrefs]             = useState(null);
  const [deleting, setDeleting]       = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [error, setError]             = useState(null);

  useEffect(() => {
    getSubscribed().then(setSubscribed).catch(() => {});
    apiFetch("/settings/notifications").then(r => setPrefs(r.prefs)).catch(() => setPrefs({}));
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

  const deleteAccount = async () => {
    if (confirmText !== "DELETE") return;
    setDeleting(true); setError(null);
    try {
      await apiFetch("/account", { method: "DELETE" });
      await supabase.auth.signOut();
    } catch (e) {
      setError(e.message);
      setDeleting(false);
    }
  };

  const rowStyle = { display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderTop: "1px solid var(--border)" };

  return (
    <div style={{ maxWidth: 560, margin: "0 auto", padding: "0 16px 24px", display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Contact */}
      <div style={card}>
        <div style={{ ...cardHeader, background: "var(--off)" }}>
          <span style={{ fontSize: 14, fontWeight: 700 }}>Contact the developer</span>
        </div>
        <div style={{ display: "flex", gap: 14, padding: "16px", justifyContent: "center" }}>
          {[
            { href: LINKS.email,    title: "Email",    inner: <Icon n="mail" size={20} style={{ color: "var(--teal)" }} /> },
            { href: LINKS.linkedin, title: "LinkedIn", inner: <LinkedInLogo /> },
            { href: LINKS.github,   title: "GitHub",   inner: <GitHubLogo /> },
          ].map(l => (
            <a key={l.title} href={l.href} target="_blank" rel="noopener noreferrer" title={l.title}
              style={{ width: 48, height: 48, borderRadius: "50%", background: "var(--off)", border: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text)" }}>
              {l.inner}
            </a>
          ))}
        </div>
      </div>

      {/* Notifications */}
      <div style={card}>
        <div style={{ ...cardHeader, background: "var(--off)" }}>
          <span style={{ fontSize: 14, fontWeight: 700 }}>Notifications</span>
        </div>

        <div style={{ ...rowStyle, borderTop: "none" }}>
          <Icon n={subscribed ? "notifications_active" : "notifications"} size={20} style={{ color: "var(--teal)", flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 700 }}>Notifications on this device</div>
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
              {permission === "denied"
                ? "Blocked in your browser/system settings"
                : subscribed ? "Enabled" : "Turn on to receive any notifications"}
            </div>
          </div>
          {pushLoading ? <Spin size={18} /> : (
            <Toggle on={subscribed} onChange={toggleMaster} disabled={permission === "denied" || !pushSupported()} />
          )}
        </div>

        {NOTIF_TYPES.map(t => (
          <div key={t.key} style={rowStyle}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: subscribed ? "var(--text)" : "var(--muted)" }}>{t.label}</div>
              <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{t.desc}</div>
            </div>
            {prefs === null ? <Spin size={14} /> : (
              <Toggle on={!!prefs[t.key]} onChange={() => togglePref(t.key)} disabled={!subscribed} />
            )}
          </div>
        ))}
      </div>

      {/* Danger zone */}
      <div style={{ ...card, border: "1px solid var(--danger)" }}>
        <div style={{ ...cardHeader, background: "var(--danger-lt)" }}>
          <span style={{ fontSize: 14, fontWeight: 700, color: "var(--danger)" }}>Danger zone</span>
        </div>
        <div style={{ padding: 16 }}>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6, marginBottom: 12 }}>
            Permanently delete your account: your food log, goals, folders, meal templates,
            scan history and login. This cannot be undone.
          </div>
          <button onClick={() => { setConfirmOpen(true); setConfirmText(""); }}
            style={{ padding: "10px 16px", background: "var(--danger)", color: "var(--on-danger)", border: "none", borderRadius: 10, fontSize: 13, fontWeight: 700, cursor: "pointer" }}>
            Delete my account
          </button>
        </div>
      </div>

      {error && (
        <div style={{ background: "var(--danger-lt)", border: "1px solid var(--danger)", borderRadius: 10, padding: "10px 14px", fontSize: 13, color: "var(--danger)" }}>
          {error}
        </div>
      )}

      {/* Delete confirmation modal */}
      {confirmOpen && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)", zIndex: 80, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
          <div style={{ background: "var(--surface)", borderRadius: 20, padding: 24, width: "100%", maxWidth: 400 }}>
            <div style={{ fontSize: 17, fontWeight: 800, color: "var(--danger)" }}>Delete account?</div>
            <div style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.6, margin: "10px 0 14px" }}>
              Everything goes: log entries, goals, folders, templates, scan history and your login.
              Type <strong style={{ color: "var(--danger)" }}>DELETE</strong> to confirm.
            </div>
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
                {deleting ? <Spin size={14} color="white" /> : "Delete forever"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
