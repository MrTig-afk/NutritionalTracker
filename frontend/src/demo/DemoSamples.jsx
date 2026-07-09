/**
 * DemoSamples.jsx — "try this label" panel for the PUBLIC DEMO build.
 *
 * Rendered by ScanTab only when the build runs with VITE_DEMO=1 (the mount is
 * behind a statically-false import.meta.env gate in normal builds, so this
 * module and the bundled label images are tree-shaken out of the real app).
 * Tapping a sample loads its synthetic label image into the normal
 * upload → crop → analyze flow, so phone visitors don't need their own photo.
 */
import React from "react";
import granolaUrl from "./labels/morning-field-granola.png";
import yoghurtUrl from "./labels/cloud-nine-yoghurt.png";
import barUrl from "./labels/peak-trail-bar.png";

const SAMPLES = [
  { url: granolaUrl, file: "morning-field-granola.png", label: "Granola" },
  { url: yoghurtUrl, file: "cloud-nine-yoghurt.png", label: "Yoghurt" },
  { url: barUrl, file: "peak-trail-bar.png", label: "Protein bar" },
];

export default function DemoSamples({ onPick }) {
  const [busy, setBusy] = React.useState(null);

  const pick = async (sample) => {
    if (busy) return;
    setBusy(sample.file);
    try {
      const blob = await fetch(sample.url).then(r => r.blob()); // bundled asset, same-origin
      onPick(new File([blob], sample.file, { type: "image/png" }));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 20, background: "var(--white)", padding: "16px 18px", display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text)" }}>No label handy? Try a sample</div>
      <div style={{ display: "flex", gap: 10 }}>
        {SAMPLES.map(s => (
          <button key={s.file} onClick={() => pick(s)} disabled={!!busy}
            style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6, padding: 8, borderRadius: 14, border: "1px solid var(--border)", background: "var(--off)", cursor: "pointer", opacity: busy && busy !== s.file ? 0.5 : 1 }}>
            <img src={s.url} alt={`Sample nutrition label: ${s.label}`} style={{ width: "100%", aspectRatio: "3 / 5", objectFit: "cover", borderRadius: 8, border: "1px solid var(--border)" }} />
            <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>{busy === s.file ? "Loading…" : s.label}</span>
          </button>
        ))}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--muted)", lineHeight: 1.4 }}>
        Synthetic labels, invented brands. In this demo any image "scans" to a sample result — the real app reads your actual label with AI vision.
      </div>
    </div>
  );
}
