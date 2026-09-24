import React, { useEffect, useState } from "react";
import { supabase } from "../lib/api";
import { Icon, Spin, Spark } from "./Icon";

// H4 · Allow (design/userflow.artifact.html, "H4 · Allow"). Reached at
// /?authorization_id=... from claude.ai; see App.jsx AUTHZ_ID.
const CONNECTOR_ORIGINS = ["https://claude.ai", "https://claude.com"];

// React StrictMode runs effects twice, and a reload does too. A second GET on
// an auto-approved id returns a 400 (supabase/auth#2820), so every effect run
// for the same id shares this one promise.
const detailsCalls = new Map();

function detailsFor(authorizationId) {
  if (!detailsCalls.has(authorizationId)) {
    detailsCalls.set(authorizationId, supabase.auth.oauth.getAuthorizationDetails(authorizationId));
  }
  return detailsCalls.get(authorizationId);
}

const errorView = (message) => (
  <div style={{ minHeight: "100dvh", background: "var(--bg)", display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center", padding: "24px 20px", gap: 14, textAlign: "center" }}>
    <div style={{ fontSize: 13, color: "var(--text)", maxWidth: 320, lineHeight: 1.5 }}>{message}</div>
    <a href="/" style={{ fontSize: 13, fontWeight: 700, color: "var(--accent)" }}>Open NutriScan</a>
  </div>
);

export default function AllowPage({ authorizationId, email, online, onDone }) {
  const [view, setView] = useState(authorizationId ? "loading" : "expired");
  const [busy, setBusy] = useState(null); // "allow" | "deny" | null

  useEffect(() => {
    if (!authorizationId) return;
    let live = true;
    detailsFor(authorizationId).then(({ data, error }) => {
      if (!live) return;
      if (error) {
        console.error("oauth details failed", error.status, error.code, error.message);
        onDone();
        setView(error.status === 404 ? "expired" : "failed");
        return;
      }
      if (data.redirect_url && !data.authorization_id) {
        // Consent was already given: follow it, and stay on the spinner.
        onDone();
        window.location.assign(data.redirect_url);
        return;
      }
      if (!CONNECTOR_ORIGINS.includes(new URL(data.redirect_uri).origin)) {   // a bad or missing URI throws: .catch
        onDone();
        setView("failed");
        return;
      }
      setView("ready");
    }).catch((e) => {
      // A thrown call (not an {error} result) must not leave the spinner up.
      if (!live) return;
      console.error("oauth details failed", e);
      onDone();
      setView("failed");
    });
    return () => { live = false; };
  }, [authorizationId, onDone]);

  const respond = async (action) => {
    setBusy(action);
    onDone();
    const call = action === "allow" ? supabase.auth.oauth.approveAuthorization : supabase.auth.oauth.denyAuthorization;
    try {
      const { error } = await call(authorizationId);
      if (error) {
        console.error("oauth details failed", error.status, error.code, error.message);
        setView("failed");
      }
    } catch (e) {
      // A thrown call (not an {error} result) must not leave both buttons spinning.
      console.error("oauth details failed", e);
      setView("failed");
    }
    // No error: supabase-js itself navigates the tab to the returned redirect_url.
  };

  if (view === "loading") {
    return (
      <div style={{ minHeight: "100dvh", background: "var(--bg)", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Spin size={36} />
      </div>
    );
  }

  if (view === "expired") return errorView("This connection request has expired. Start again from claude.ai.");
  if (view === "failed") return errorView("Couldn't connect Claude. Try again from claude.ai.");

  return (
    <div style={{ minHeight: "100dvh", background: "var(--bg)", display: "flex", alignItems: "center",
      justifyContent: "center", padding: "24px 20px" }}>
      <div style={{ width: "100%", maxWidth: 400, background: "var(--surface)", border: "1px solid var(--border)",
        borderRadius: 14, padding: "20px 20px", display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
          <span style={{ width: 34, height: 34, borderRadius: 10, border: "1.5px solid var(--border)",
            display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Spark size={22} color="var(--claude)" />
          </span>
          <Icon n="sync_alt" style={{ color: "var(--muted)" }} />
          <span style={{ width: 34, height: 34, borderRadius: 10, background: "var(--teal)", border: "none",
            display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Icon n="nutrition" size={20} style={{ color: "var(--mint)" }} />
          </span>
        </div>
        <div style={{ fontSize: 16, fontWeight: 800, textAlign: "center", color: "var(--text)" }}>
          Claude wants to connect to NutriScan
        </div>
        <div style={{ fontSize: 13, color: "var(--muted)", textAlign: "center" }}>
          NutriScan account: {email}
        </div>
        <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text2)" }}>Claude will be able to:</div>
        <div style={{ fontSize: 13, color: "var(--text)" }}>· Read your food log, goals, meal templates and Library</div>
        <div style={{ fontSize: 13, color: "var(--text)" }}>· Add, change or delete log entries and log templates, after showing you first and you say yes</div>
        <div style={{ fontSize: 13, color: "var(--text)" }}>· Save foods to your Library</div>
        <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text2)" }}>It can't:</div>
        <div style={{ fontSize: 13, color: "var(--text)" }}>· Change your goals, notifications or account, or delete your account</div>
        <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
          <button onClick={() => respond("deny")} disabled={!!busy}
            style={{ flex: 1, padding: "10px", borderRadius: 10, border: "1.5px solid var(--border)",
              background: "var(--surface)", color: "var(--text)", fontSize: 13, fontWeight: 700,
              cursor: busy ? "not-allowed" : "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
            {busy === "deny" ? <Spin size={18} /> : "Deny"}
          </button>
          <button onClick={() => respond("allow")} disabled={!!busy || !online}
            style={{ flex: 1, padding: "10px", borderRadius: 10, border: "none", background: "var(--teal)",
              color: "white", fontSize: 13, fontWeight: 700, cursor: (busy || !online) ? "not-allowed" : "pointer",
              opacity: !online ? 0.55 : 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
            {busy === "allow" ? <Spin size={18} color="white" /> : "Allow"}
          </button>
        </div>
        <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center" }}>
          Disconnect any time in Settings → Connected apps.
        </div>
      </div>
    </div>
  );
}
