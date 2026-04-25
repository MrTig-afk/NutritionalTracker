import React, { useState, useEffect, useCallback } from "react";
import { supabase } from "./lib/api";
import { PALETTE_CSS } from "./styles";
import { Icon, Spin } from "./components/Icon";
import AddToLogModal from "./components/AddToLogModal";
import EditLogModal from "./components/EditLogModal";
import ChatAssistant from "./components/ChatAssistant";
import LoginScreen from "./components/LoginScreen";
import ScanTab from "./tabs/ScanTab";
import LibraryTab from "./tabs/LibraryTab";
import TrackerTab from "./tabs/TrackerTab";
import TrendsTab from "./tabs/TrendsTab";

const TABS = [
  { id: "scan",    label: "Scan",    icon: "document_scanner" },
  { id: "library", label: "Library", icon: "folder"           },
  { id: "tracker", label: "Tracker", icon: "bar_chart"        },
  { id: "trends",  label: "Trends",  icon: "show_chart"       },
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
  const [scanBusy, setScanBusy]           = useState(false);
  const [showIOSBanner, setShowIOSBanner] = useState(
    () => isIOSNotInstalled() && !localStorage.getItem("ios-banner-dismissed")
  );

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });
    return () => subscription.unsubscribe();
  }, []);

  const handleAddToLog  = useCallback((item) => { setAddToLogItem(item); }, []);
  const handleLogAdded  = useCallback(() => { setLogRefreshKey(k => k + 1); }, []);
  const handleEditEntry = useCallback((entry) => { setEditLogItem(entry); }, []);

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
          <div style={{ background: "#004E56", padding: "10px 16px", display: "flex", alignItems: "center", gap: 10, zIndex: 39 }}>
            <Icon n="ios_share" size={18} style={{ color: "var(--mint)", flexShrink: 0 }} />
            <span style={{ flex: 1, fontSize: 13, color: "var(--mint)", lineHeight: 1.4 }}>
              Tap <strong>Share</strong> → <strong>Add to Home Screen</strong> to install NutriScan
            </span>
            <button onClick={dismissIOSBanner} style={{ background: "none", border: "none", cursor: "pointer", padding: 4, color: "rgba(174,246,199,0.6)", flexShrink: 0 }}>
              <Icon n="close" size={18} />
            </button>
          </div>
        )}

        {/* Top App Bar */}
        <div style={{ position: "sticky", top: 0, zIndex: 40, background: "var(--teal)", paddingTop: "calc(12px + env(safe-area-inset-top, 0px))", paddingBottom: "12px", paddingLeft: "20px", paddingRight: "20px", display: "flex", alignItems: "center", justifyContent: "space-between", boxShadow: "0 2px 8px rgba(0,109,119,0.25)" }}>
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
            {avatarUrl
              ? <img src={avatarUrl} alt="avatar" onError={e => { e.target.style.display = "none"; e.target.nextSibling.style.display = "flex"; }} style={{ width: 34, height: 34, borderRadius: "50%", objectFit: "cover", border: "2px solid rgba(174,246,199,0.5)" }} />
              : null}
            <div style={{ width: 34, height: 34, borderRadius: "50%", background: "rgba(174,246,199,0.2)", border: "2px solid rgba(174,246,199,0.4)", display: avatarUrl ? "none" : "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 800, color: "var(--mint)" }}>{userInitial}</div>
            <button onClick={() => supabase.auth.signOut()} style={{ fontSize: 12, padding: "5px 12px", background: "rgba(174,246,199,0.12)", border: "1px solid rgba(174,246,199,0.25)", borderRadius: 20, color: "var(--mint)", cursor: "pointer", fontWeight: 600 }}>
              Sign out
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="ns-content" style={{ flex: 1, width: "100%", margin: "0 auto", padding: "20px 32px" }}>
          <div style={{ display: activeMainTab === "scan" ? "block" : "none" }}>
            <ScanTab onAddToLog={handleAddToLog} onBusyChange={setScanBusy} />
          </div>
          {activeMainTab === "library" && <LibraryTab key={libraryMountKey} onAddToLog={handleAddToLog} />}
          <div style={{ display: activeMainTab === "tracker" ? "block" : "none" }}>
            <TrackerTab refreshKey={logRefreshKey} onEditEntry={handleEditEntry} />
          </div>
          {activeMainTab === "trends" && <TrendsTab />}
        </div>

        {/* Bottom Navigation */}
        <div className="ns-bottom-nav" style={{ position: "fixed", bottom: 0, left: 0, right: 0, zIndex: 40, background: "var(--surface)", borderTop: "1px solid var(--border)", paddingTop: 6, paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
          {TABS.map(tab => {
            const active = activeMainTab === tab.id;
            return (
              <button key={tab.id} onClick={() => handleTabChange(tab.id)}
                style={{ flex: 1, height: 68, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4, border: "none", background: "none", cursor: "pointer", padding: 0, color: active ? "var(--teal)" : "var(--muted)", transition: "color 0.15s" }}>
                <div style={{ width: 56, height: 28, borderRadius: 14, background: active ? "var(--teal-lt)" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", transition: "background 0.15s" }}>
                  <Icon n={tab.icon} size={22} style={{ color: active ? "var(--teal)" : "var(--muted)", fontVariationSettings: active ? "'FILL' 1" : "'FILL' 0" }} />
                </div>
                <span style={{ fontSize: 11, fontWeight: active ? 700 : 500, letterSpacing: "0.2px" }}>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <ChatAssistant hidden={!!(addToLogItem || editLogItem || scanBusy)} />
    </>
  );
}
