import React, { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, supabase } from "../lib/api";
import { confirm } from "../lib/confirm";
import { card, cardHeader, ghostBtn, errorBanner, inputStyle } from "../styles";
import { Icon, Spin, Spark } from "./Icon";

// H1 (empty) and H6 (list) of the master Artifact v7. Owner-only: a 403 renders nothing, like the Admin card.
const MCP_ADDRESS = "https://nutritionaltracker.onrender.com/mcp";
const OWNER_KEY = "connectedAppsSeen";
const recall = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const remember = (k, v) => { try { v === null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* private mode */ } };

const MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ");
const day = (iso) => { const d = new Date(iso); return `${d.getDate()} ${MONTHS[d.getMonth()]}`; };   // "24 Sep" as in H6 (en-AU says "Sept")
function ago(iso) {
  if (!iso) return "never";
  const min = Math.round((Date.now() - new Date(iso)) / 60000);
  if (min < 1) return "just now";
  if (min < 60) return `${min} min ago`;
  if (min < 1440) return `${Math.round(min / 60)} h ago`;
  return min < 2880 ? "yesterday" : `${Math.round(min / 1440)} days ago`;
}

function Steps() {
  const [copied, setCopied] = useState(false);
  const copy = () => navigator.clipboard.writeText(MCP_ADDRESS).then(() => setCopied(true)).catch(() => {});
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 13, color: "var(--text)" }}>
      <div>1. In claude.ai open Customize → Connectors → Add custom connector.</div>
      <div>2. Paste this address:</div>
      <div style={{ fontFamily: "monospace", fontSize: 12, background: "var(--off)", borderRadius: 8, padding: "6px 8px", wordBreak: "break-all" }}>{MCP_ADDRESS}</div>
      <button onClick={copy} style={ghostBtn}><Icon n={copied ? "check" : "content_copy"} size={14} /> {copied ? "Copied" : "Copy address"}</button>
      <div>3. Sign in here when Claude asks, then tap Allow.</div>
    </div>
  );
}

export default function ConnectedAppsCard() {
  const [apps, setApps] = useState(null);        // null = hidden (loading, or not the owner)
  const [loadErr, setLoadErr] = useState(false);
  const [busy, setBusy] = useState(null);        // id being disconnected
  const [failed, setFailed] = useState(null);    // id whose disconnect failed
  const [editing, setEditingState] = useState(null);  // { id, name }
  // The blur of an input that is being removed runs its old handler: it reads this ref, never a stale closure,
  // so Escape (then blur) saves nothing and Enter (then blur) saves once.
  const editRef = useRef(null);
  const setEditing = (v) => { editRef.current = v; setEditingState(v); };
  const [showHow, setShowHow] = useState(false);

  const load = useCallback(() => {
    setLoadErr(false);
    apiFetch("/settings/connected-apps").then((rows) => {
      remember(OWNER_KEY, "1");
      setApps(rows);
    }).catch((e) => {
      if (e.status === 403) { remember(OWNER_KEY, null); return; }   // not the owner: the card never renders
      // Offline or a server error: say so only where this device has seen the card before, never to other accounts.
      if (recall(OWNER_KEY)) { setApps((a) => a || []); setLoadErr(true); }
    });
  }, []);
  useEffect(load, [load]);

  const disconnect = async (app) => {
    const ok = await confirm("It loses access on its next request. Entries it logged stay.",
      { title: "Disconnect Claude?", okLabel: "Disconnect" });
    if (!ok) return;
    setBusy(app.id); setFailed(null);
    try {
      // Supabase's grant FIRST: once it is gone a reconnect must pass the Allow page, which revives our row.
      // (Ours first could leave a refused row behind a live grant: claude.ai would reconnect without asking
      // and be refused every time.) 404 = an earlier try already revoked it.
      const { error } = await supabase.auth.oauth.revokeGrant({ clientId: app.client_id });
      if (error && error.status !== 404) throw error;
      // Then ours: this is what refuses Claude's very next call. 404 = an earlier try already did it.
      await apiFetch(`/settings/connected-apps/${app.id}`, { method: "DELETE" }).catch((e) => { if (e.status !== 404) throw e; });
      setApps((a) => a.filter((x) => x.id !== app.id));
    } catch (e) {
      console.error("disconnect failed", e);
      setFailed(app.id);
    } finally {
      setBusy(null);
    }
  };

  const saveName = async () => {
    if (!editRef.current) return;                // already saved or cancelled
    const { id, name: typed } = editRef.current;
    setEditing(null);
    const name = typed.trim();
    if (!name || name.length > 40 || name === apps.find((x) => x.id === id)?.name) return;
    try {
      const row = await apiFetch(`/settings/connected-apps/${id}`, { method: "PATCH", body: JSON.stringify({ name }) });
      setApps((a) => a.map((x) => (x.id === row.id ? row : x)));
    } catch (e) { console.error("rename failed", e); }   // the old name stays on screen
  };

  if (apps === null) return null;
  return (
    <div style={card}>
      <div style={{ ...cardHeader, justifyContent: "flex-start" }}>
        <Spark size={16} color="var(--claude)" />
        <span style={{ fontSize: 14, fontWeight: 700, flex: 1, marginLeft: 8 }}>Connected apps</span>
        {apps.length > 0 && <span style={{ background: "var(--teal-lt)", color: "var(--accent)", borderRadius: 20, padding: "2px 8px", fontSize: 11, fontWeight: 700 }}>{apps.length}</span>}
      </div>
      <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
        {loadErr && <button onClick={load} style={{ ...errorBanner, textAlign: "left", cursor: "pointer" }}>Couldn't load connected apps. Try again.</button>}
        {!loadErr && apps.length === 0 && (
          <>
            <div style={{ fontSize: 13, color: "var(--muted)" }}>Nothing connected yet.</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text2)" }}>Connect Claude chat</div>
            <Steps />
          </>
        )}
        {apps.map((app) => (
          <div key={app.id} style={{ display: "flex", flexDirection: "column", gap: 6, borderTop: "1px solid var(--off)", paddingTop: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                {editing?.id === app.id
                  ? <input autoFocus maxLength={40} value={editing.name} style={inputStyle} aria-label="Connection name"
                      onChange={(e) => setEditing({ ...editRef.current, name: e.target.value })}
                      onBlur={saveName} onKeyDown={(e) => { if (e.key === "Enter") saveName(); if (e.key === "Escape") setEditing(null); }} />
                  : <button onClick={() => setEditing({ id: app.id, name: app.name })} aria-label={`Rename ${app.name}`}
                      style={{ background: "none", border: "none", padding: 0, fontSize: 13, fontWeight: 700, color: "var(--text)", cursor: "pointer" }}>{app.name}</button>}
                <div style={{ fontSize: 11, color: "var(--muted)" }}>Connected {day(app.connected_at)} · last used {ago(app.last_used_at)}</div>
              </div>
              <button onClick={() => disconnect(app)} disabled={busy === app.id}
                style={{ ...ghostBtn, color: "var(--danger)", borderColor: "var(--danger)" }}>
                {busy === app.id ? <Spin size={14} /> : "Disconnect"}
              </button>
            </div>
            {failed === app.id && <div style={errorBanner}>Couldn't disconnect. Try again.</div>}
          </div>
        ))}
        {apps.length > 0 && (
          <>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>Each Claude account you connected shows here on its own.</div>
            <button onClick={() => setShowHow((v) => !v)} style={{ background: "none", border: "none", padding: 0, fontSize: 12, fontWeight: 700, color: "var(--accent)", cursor: "pointer", textAlign: "left" }}>How to connect Claude chat</button>
            {showHow && <Steps />}
          </>
        )}
      </div>
    </div>
  );
}
