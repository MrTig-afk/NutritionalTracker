import React, { useState } from "react";
import { supabase, API_URL } from "../lib/api";
import { Icon, Spin } from "./Icon";

export default function LoginScreen() {
  const [email, setEmail]                   = useState("");
  const [sending, setSending]               = useState(false);
  const [sent, setSent]                     = useState(false);
  const [error, setError]                   = useState(null);
  const [googleConflict, setGoogleConflict] = useState(false);
  const [code, setCode]                     = useState("");
  const [verifying, setVerifying]           = useState(false);

  const signInWithGoogle = () =>
    supabase.auth.signInWithOAuth({ provider: "google", options: { redirectTo: window.location.origin } });

  const handleSubmit = async () => {
    if (!email.trim()) return;
    const normalizedEmail = email.trim().toLowerCase();
    setSending(true); setError(null); setGoogleConflict(false);
    try {
      try {
        const res = await fetch(`${API_URL}/check-provider?email=${encodeURIComponent(normalizedEmail)}`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
          const { has_google } = await res.json();
          if (has_google) {
            setGoogleConflict(true);
            signInWithGoogle();
            return;
          }
        }
      } catch (_) {}

      const { error: otpError } = await supabase.auth.signInWithOtp({ email: normalizedEmail });
      if (otpError) throw new Error(otpError.message);
      setSent(true);
    } catch (e) {
      setError(e.message);
    } finally { setSending(false); }
  };

  // Verify the emailed 6-digit code in place — no redirect, so the session is
  // created in this exact context (installed PWA or any browser).
  const handleVerifyCode = async () => {
    const token = code.trim();
    if (token.length < 6) return;
    setVerifying(true); setError(null);
    try {
      const { error: vErr } = await supabase.auth.verifyOtp({
        email: email.trim().toLowerCase(), token, type: "email",
      });
      if (vErr) throw new Error(vErr.message);
      // Success: onAuthStateChange in App.jsx takes over.
    } catch (e) {
      setError(e.message);
      setVerifying(false);
    }
  };

  const inputStyle = {
    width: "100%", padding: "12px 14px",
    background: "var(--off)", border: "1.5px solid var(--border)",
    borderRadius: 12, fontSize: 15, color: "var(--text)",
  };

  return (
    <div style={{ minHeight: "100dvh", background: "var(--bg)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "24px 20px" }}>
      <div style={{ width: "100%", maxWidth: 400, display: "flex", flexDirection: "column", gap: 28 }}>

        <div style={{ textAlign: "center" }}>
          <div style={{ width: 64, height: 64, borderRadius: 20, background: "var(--teal)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 18px", boxShadow: "0 8px 24px rgba(0,109,119,0.35)" }}>
            <Icon n="nutrition" size={30} style={{ color: "var(--mint)" }} />
          </div>
          <div style={{ fontSize: 30, fontWeight: 800, color: "var(--text)", letterSpacing: "-0.8px" }}>NutriScan</div>
          <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 6 }}>Track your nutrition with ease</div>
        </div>

        <div style={{ background: "var(--surface)", borderRadius: 24, padding: 28, display: "flex", flexDirection: "column", gap: 18, boxShadow: "0 2px 16px rgba(0,0,0,0.08), 0 0 0 1px var(--border)" }}>
          {!sent ? (
            <>
              <div>
                <div style={{ fontSize: 18, fontWeight: 700, color: "var(--text)" }}>Welcome back</div>
                <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 5, lineHeight: 1.5 }}>Enter your email to receive a sign-in code.</div>
              </div>

              <div>
                <label style={{ fontSize: 11, color: "var(--muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.6px", display: "block", marginBottom: 7 }}>Email address</label>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSubmit()}
                  placeholder="you@example.com"
                  style={inputStyle}
                  autoFocus
                />
              </div>

              {googleConflict && (
                <div style={{ background: "var(--teal-lt, #E0F4F5)", border: "1px solid var(--teal)", borderRadius: 10, padding: "10px 14px", fontSize: 13, color: "var(--teal)", display: "flex", alignItems: "center", gap: 8 }}>
                  <Icon n="info" size={16} style={{ color: "var(--teal)", flexShrink: 0 }} />
                  An account with this email already exists. Redirecting you to sign in with Google…
                </div>
              )}

              {error && (
                <div style={{ background: "var(--danger-lt, #FFDAD6)", border: "1px solid var(--danger)", borderRadius: 10, padding: "10px 14px", fontSize: 13, color: "var(--danger)" }}>
                  {error}
                </div>
              )}

              <button
                onClick={handleSubmit}
                disabled={sending || !email.trim()}
                style={{ width: "100%", padding: "14px", background: "var(--teal)", color: "white", border: "none", borderRadius: 14, fontSize: 15, fontWeight: 700, cursor: sending || !email.trim() ? "not-allowed" : "pointer", opacity: sending || !email.trim() ? 0.55 : 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, letterSpacing: "0.2px" }}>
                {sending ? <Spin size={18} color="white" /> : <Icon n="send" size={18} style={{ color: "white" }} />}
                {sending ? "Sending…" : "Send Sign-in Code"}
              </button>

              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
                <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 600 }}>or</span>
                <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
              </div>

              <button
                onClick={signInWithGoogle}
                style={{ width: "100%", padding: "13px", background: "var(--surface)", color: "var(--text)", border: "1.5px solid var(--border)", borderRadius: 14, fontSize: 14, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
                <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
                  <path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 0 0 2.38-5.88c0-.57-.05-.66-.15-1.18z"/>
                  <path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2.04a4.8 4.8 0 0 1-7.18-2.54H1.83v2.07A8 8 0 0 0 8.98 17z"/>
                  <path fill="#FBBC05" d="M4.5 10.48A4.8 4.8 0 0 1 4.5 7.52V5.45H1.83a8 8 0 0 0 0 7.1l2.67-2.07z"/>
                  <path fill="#EA4335" d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 0 0 1.83 5.45L4.5 7.52A4.8 4.8 0 0 1 8.98 4.18z"/>
                </svg>
                Continue with Google
              </button>
            </>
          ) : (
            <div style={{ textAlign: "center", padding: "12px 0" }}>
              <div style={{ width: 56, height: 56, borderRadius: "50%", background: "var(--teal-lt)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 18px" }}>
                <Icon n="mark_email_read" size={28} style={{ color: "var(--teal)" }} />
              </div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "var(--text)" }}>Check your inbox</div>
              <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 10, lineHeight: 1.7 }}>
                We sent a sign-in email to<br /><strong style={{ color: "var(--text)" }}>{email}</strong><br />
                Enter the 6-digit code from it below — or tap the link in the email.
              </div>

              <input
                type="text"
                inputMode="numeric"
                autoComplete="one-time-code"
                maxLength={6}
                value={code}
                onChange={e => setCode(e.target.value.replace(/\D/g, ""))}
                onKeyDown={e => e.key === "Enter" && handleVerifyCode()}
                placeholder="000000"
                autoFocus
                style={{ ...inputStyle, marginTop: 18, textAlign: "center", fontSize: 22, fontWeight: 700, letterSpacing: "8px" }}
              />

              {error && (
                <div style={{ background: "var(--danger-lt, #FFDAD6)", border: "1px solid var(--danger)", borderRadius: 10, padding: "10px 14px", fontSize: 13, color: "var(--danger)", marginTop: 12, textAlign: "left" }}>
                  {error}
                </div>
              )}

              <button
                onClick={handleVerifyCode}
                disabled={verifying || code.trim().length < 6}
                style={{ width: "100%", marginTop: 14, padding: "14px", background: "var(--teal)", color: "white", border: "none", borderRadius: 14, fontSize: 15, fontWeight: 700, cursor: verifying || code.trim().length < 6 ? "not-allowed" : "pointer", opacity: verifying || code.trim().length < 6 ? 0.55 : 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                {verifying ? <Spin size={18} color="white" /> : <Icon n="login" size={18} style={{ color: "white" }} />}
                {verifying ? "Verifying…" : "Sign In"}
              </button>

              <button onClick={() => { setSent(false); setEmail(""); setCode(""); setError(null); }} style={{ marginTop: 16, fontSize: 13, color: "var(--teal)", background: "none", border: "none", cursor: "pointer", fontWeight: 700, textDecoration: "underline" }}>
                Use a different email
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
