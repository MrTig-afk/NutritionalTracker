import React, { useState, useEffect, useCallback } from "react";
import { Analytics } from "@vercel/analytics/react";
import { supabase } from "./lib/api";
import { PALETTE_CSS } from "./styles";
import { CHANGELOG_VERSION } from "./version";
import { Icon, Spin } from "./components/Icon";
import AddToLogModal from "./components/AddToLogModal";
import EditLogModal from "./components/EditLogModal";
import ChatAssistant from "./components/ChatAssistant";
import LoginScreen from "./components/LoginScreen";
import ScanTab from "./tabs/ScanTab";
import LibraryTab from "./tabs/LibraryTab";
import TrackerTab from "./tabs/TrackerTab";
import TrendsTab from "./tabs/TrendsTab";
import SettingsTab from "./tabs/SettingsTab";

const TABS = [
  { id: "scan",     label: "Scan",     icon: "document_scanner" },
  { id: "library",  label: "Library",  icon: "folder"           },
  { id: "tracker",  label: "Tracker",  icon: "bar_chart"        },
  { id: "trends",   label: "Trends",   icon: "show_chart"       },
  { id: "ai",       label: "AI",       icon: "nutrition"        },
  { id: "settings", label: "Settings", icon: "settings"         },
];

const isIOSNotInstalled = () => {
  const isIOS = /iPhone|iPad|iPod/.test(navigator.userAgent);
  const isStandalone = window.navigator.standalone === true;
  return isIOS && !isStandalone;
};

export default function App() {
  const [session, setSession]             = useState(undefined);
  const [activeMainTab, setActiveMainTab] = useState("scan");
  const [addToLogItem, setAddToLogItem]   = useState(null);
  const [logRefreshKey, setLogRefreshKey] = useState(0);
  const [libraryMountKey, setLibraryMountKey] = useState(0);
  const [editLogItem, setEditLogItem]     = useState(null);
  const [updateReady, setUpdateReady]   = useState(() => !!window.__swUpdateReady);
  const [showChangelog, setShowChangelog] = useState(false);
  const [upcomingChangelog, setUpcomingChangelog] = useState([]);
  const [showIOSBanner, setShowIOSBanner] = useState(
    () => isIOSNotInstalled() && !localStorage.getItem("ios-banner-dismissed")
  );
  const [theme, setTheme] = useState(() =>
    localStorage.getItem("theme")
    || (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#1B2423" : "#006D77");
  }, [theme]);

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });
    return () => subscription.unsubscribe();
  }, []);

  // Boot fallback: never leave the user on the spinner. If the auth server is
  // slow/unreachable, fall through to the login screen after 6s.
  useEffect(() => {
    let cancelled = false;
    const settle = (value) => { if (!cancelled) setSession(s => (s === undefined ? value : s)); };
    const timer = setTimeout(() => settle(null), 6000);
    supabase.auth.getSession()
      .then(({ data }) => settle(data?.session ?? null))
      .catch(() => settle(null));
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  useEffect(() => {
    const handler = () => {
      setUpdateReady(true);
      // Versioned notes: show only entries newer than the build we're running,
      // so the prompt always matches exactly what the pending update contains.
      fetch("/changelog.v2.json", { cache: "reload" })
        .then(r => r.json())
        .then(items => setUpcomingChangelog(
          (Array.isArray(items) ? items : [])
            .filter(i => typeof i === "object" && i.v > CHANGELOG_VERSION)
            .map(i => i.text)
        ))
        .catch(() => {});
    };
    window.addEventListener('sw-update-ready', handler);
    return () => window.removeEventListener('sw-update-ready', handler);
  }, []);

  const handleAddToLog  = useCallback((item) => { setAddToLogItem(item); }, []);
  const handleLogAdded  = useCallback(() => { setLogRefreshKey(k => k + 1); }, []);
  const handleEditEntry = useCallback((entry) => { setEditLogItem(entry); }, []);

  const handleUpdate = () => {
    navigator.serviceWorker.ready.then(reg => {
      if (reg.waiting) reg.waiting.postMessage({ type: 'SKIP_WAITING' });
      window.location.reload();
    });
  };

  // If an update is ready while we're still stuck on the boot spinner, the
  // running build is likely broken — apply the new one immediately (there is
  // no in-app state to lose before login).
  useEffect(() => {
    if (updateReady && session === undefined) handleUpdate();
  }, [updateReady, session]);

  const dismissIOSBanner = () => {
    localStorage.setItem("ios-banner-dismissed", "1");
    setShowIOSBanner(false);
  };

  const handleTabChange = useCallback((tabId) => {
    setActiveMainTab(tabId);
    if (tabId === "library") setLibraryMountKey(k => k + 1);
  }, []);

  if (session === undefined) {
    return (
      <>
        <style>{PALETTE_CSS}</style>
        <div style={{ minHeight: "100dvh", background: "var(--bg)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Spin size={36} />
        </div>
      </>
    );
  }

  if (!session) {
    return (
      <>
        <style>{PALETTE_CSS}</style>
        {updateReady && (
          <button onClick={handleUpdate}
            style={{ position: "fixed", top: "calc(12px + env(safe-area-inset-top, 0px))", left: "50%", transform: "translateX(-50%)", zIndex: 100, background: "var(--teal)", color: "white", border: "none", borderRadius: 20, padding: "8px 16px", fontSize: 13, fontWeight: 700, cursor: "pointer", boxShadow: "0 4px 12px rgba(0,109,119,0.4)" }}>
            Update available — tap to refresh
          </button>
        )}
        <LoginScreen />
      </>
    );
  }

  const avatarUrl   = session?.user?.user_metadata?.avatar_url;
  const userInitial = (session?.user?.email || "U")[0].toUpperCase();

  return (
    <>
      <style>{PALETTE_CSS}</style>
      <div style={{ minHeight: "100dvh", background: "var(--bg)", color: "var(--text)", display: "flex", flexDirection: "column" }}>
        {addToLogItem && <AddToLogModal item={addToLogItem} onClose={() => setAddToLogItem(null)} onAdded={handleLogAdded} />}
        {editLogItem  && <EditLogModal  entry={editLogItem}  onClose={() => setEditLogItem(null)}  onSaved={() => { setLogRefreshKey(k => k + 1); }} />}

        {showIOSBanner && (
          <div style={{ background: "var(--mint)", padding: "10px 16px", display: "flex", alignItems: "center", gap: 10, zIndex: 39 }}>
            <Icon n="ios_share" size={18} style={{ color: "#0B3D22", flexShrink: 0 }} />
            <span style={{ flex: 1, fontSize: 13, color: "#0B3D22", lineHeight: 1.4 }}>
              Tap <strong>Share</strong> → <strong>Add to Home Screen</strong> to install NutriScan
            </span>
            <button onClick={dismissIOSBanner} style={{ background: "none", border: "none", cursor: "pointer", padding: 4, color: "rgba(11,61,34,0.6)", flexShrink: 0 }}>
              <Icon n="close" size={18} />
            </button>
          </div>
        )}

        {/* Top App Bar */}
        <div style={{ position: "sticky", top: 0, zIndex: 40, background: "var(--header)", paddingTop: "calc(12px + env(safe-area-inset-top, 0px))", paddingBottom: "12px", paddingLeft: "20px", paddingRight: "20px", display: "flex", alignItems: "center", justifyContent: "space-between", boxShadow: "0 2px 8px rgba(0,0,0,0.25)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: "rgba(174,246,199,0.18)", border: "1.5px solid rgba(174,246,199,0.3)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Icon n="nutrition" size={20} style={{ color: "var(--mint)" }} />
            </div>
            <span style={{ fontSize: 20, fontWeight: 800, color: "white", letterSpacing: "-0.4px" }}>NutriScan</span>
          </div>

          <div className="ns-top-tabs" style={{ alignItems: "center", gap: 4 }}>
            {TABS.map(tab => {
              const active = activeMainTab === tab.id;
              return (
                <button key={tab.id} onClick={() => handleTabChange(tab.id)}
                  style={{ display: "flex", alignItems: "center", gap: 6, padding: "7px 16px", borderRadius: 10, border: "none", fontSize: 13, fontWeight: 600, cursor: "pointer", background: active ? "rgba(174,246,199,0.2)" : "transparent", color: active ? "var(--mint)" : "rgba(255,255,255,0.7)", transition: "all 0.15s" }}>
                  <Icon n={tab.icon} size={18} style={{ fontVariationSettings: active ? "'FILL' 1" : "'FILL' 0" }} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button onClick={() => setTheme(t => (t === "dark" ? "light" : "dark"))} title="Toggle dark mode"
              style={{ width: 34, height: 34, borderRadius: "50%", background: "rgba(174,246,199,0.12)", border: "1px solid rgba(174,246,199,0.25)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--mint)", padding: 0 }}>
              <Icon n={theme === "dark" ? "light_mode" : "dark_mode"} size={18} />
            </button>
            {avatarUrl
              ? <img src={avatarUrl} alt="avatar" onError={e => { e.target.style.display = "none"; e.target.nextSibling.style.display = "flex"; }} style={{ width: 34, height: 34, borderRadius: "50%", objectFit: "cover", border: "2px solid rgba(174,246,199,0.5)" }} />
              : null}
            <div style={{ width: 34, height: 34, borderRadius: "50%", background: "rgba(174,246,199,0.2)", border: "2px solid rgba(174,246,199,0.4)", display: avatarUrl ? "none" : "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 800, color: "var(--mint)" }}>{userInitial}</div>
            <button onClick={() => supabase.auth.signOut()} style={{ fontSize: 12, padding: "5px 12px", background: "rgba(174,246,199,0.12)", border: "1px solid rgba(174,246,199,0.25)", borderRadius: 20, color: "var(--mint)", cursor: "pointer", fontWeight: 600 }}>
              Sign out
            </button>
          </div>
        </div>

        {/* Update banner — kept out of the header so it never crowds it */}
        {updateReady && (
          <button onClick={() => setShowChangelog(true)}
            style={{ width: "100%", background: "var(--mint)", border: "none", padding: "10px 16px", display: "flex", alignItems: "center", justifyContent: "center", gap: 10, cursor: "pointer" }}>
            <Icon n="new_releases" size={18} style={{ color: "#0B3D22", flexShrink: 0 }} />
            <span style={{ fontSize: 13, color: "#0B3D22", fontWeight: 700 }}>
              Update available — tap to see what's new
            </span>
          </button>
        )}

        {/* Content */}
        <div className="ns-content" style={{ flex: 1, width: "100%", margin: "0 auto", paddingTop: 20 }}>
          <div style={{ display: activeMainTab === "scan" ? "block" : "none" }}>
            <ScanTab onAddToLog={handleAddToLog} />
          </div>
          {activeMainTab === "library" && <LibraryTab key={libraryMountKey} onAddToLog={handleAddToLog} />}
          <div style={{ display: activeMainTab === "tracker" ? "block" : "none" }}>
            <TrackerTab refreshKey={logRefreshKey} onEditEntry={handleEditEntry} />
          </div>
          {activeMainTab === "trends" && <TrendsTab />}
          {activeMainTab === "settings" && <SettingsTab />}
        </div>

        {/* Bottom Navigation — AI is a normal tab like everything else */}
        <div className="ns-bottom-nav" style={{ position: "fixed", bottom: 0, left: 0, right: 0, zIndex: 40, background: "var(--surface)", borderTop: "1px solid var(--border)", paddingTop: 6, paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
          {TABS.map(tab => {
            const active = activeMainTab === tab.id;
            return (
              <button key={tab.id} onClick={() => handleTabChange(tab.id)}
                style={{ flex: 1, height: 68, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4, border: "none", background: "none", cursor: "pointer", padding: 0, color: active ? "var(--accent)" : "var(--muted)", transition: "color 0.15s" }}>
                <div style={{ width: 56, height: 28, borderRadius: 14, background: active ? "var(--teal-lt)" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", transition: "background 0.15s" }}>
                  <Icon n={tab.icon} size={22} style={{ color: active ? "var(--accent)" : "var(--muted)", fontVariationSettings: active ? "'FILL' 1" : "'FILL' 0" }} />
                </div>
                <span style={{ fontSize: 11, fontWeight: active ? 700 : 500, letterSpacing: "0.2px" }}>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {showChangelog && (
        <div style={{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(0,0,0,0.55)", display: "flex", alignItems: "center", justifyContent: "center", padding: "0 20px" }}>
          <div style={{ background: "var(--surface)", borderRadius: 20, padding: "28px 24px", width: "100%", maxWidth: 400, boxShadow: "0 20px 60px rgba(0,0,0,0.3)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <div style={{ width: 34, height: 34, borderRadius: 10, background: "var(--teal-lt)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Icon n="new_releases" size={18} style={{ color: "var(--accent)" }} />
              </div>
              <div>
                <div style={{ fontSize: 16, fontWeight: 800, color: "var(--text)" }}>Update Available</div>
                <div style={{ fontSize: 11, color: "var(--muted)" }}>What's new in this version</div>
              </div>
            </div>
            <ul style={{ margin: "16px 0", padding: "0 0 0 18px", display: "flex", flexDirection: "column", gap: 8 }}>
              {upcomingChangelog.length > 0
                ? upcomingChangelog.map((item, i) => (
                    <li key={i} style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.45 }}>{item}</li>
                  ))
                : <li style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.45 }}>New fixes and improvements</li>
              }
            </ul>
            <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
              <button onClick={() => setShowChangelog(false)}
                style={{ flex: 1, padding: "10px", borderRadius: 12, border: "1px solid var(--border)", background: "none", fontSize: 13, fontWeight: 600, color: "var(--muted)", cursor: "pointer" }}>
                Later
              </button>
              <button onClick={handleUpdate}
                style={{ flex: 2, padding: "10px", borderRadius: 12, border: "none", background: "var(--teal)", fontSize: 13, fontWeight: 700, color: "#fff", cursor: "pointer" }}>
                Update Now
              </button>
            </div>
            <p style={{ margin: "12px 0 0", fontSize: 11, color: "var(--muted)", textAlign: "center", lineHeight: 1.5 }}>
              Choosing Later keeps your current version until you close and reopen the app.
            </p>
          </div>
        </div>
      )}

      {/* Kept mounted so the conversation survives tab switches */}
      <ChatAssistant open={activeMainTab === "ai"} />
      <Analytics />
    </>
  );
}
