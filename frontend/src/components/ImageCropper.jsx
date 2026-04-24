import React, { useState, useRef, useEffect, useCallback } from "react";
import { overlayBg } from "../styles";
import { Icon, Spin } from "./Icon";

export default function ImageCropper({ file, onConfirm, onCancel }) {
  const canvasRef = useRef(null); const containerRef = useRef(null);
  const imgRef = useRef(null); const isDragging = useRef(false);
  const dragMode = useRef(null); const dragStart = useRef({ x: 0, y: 0 });
  const cropAtStart = useRef(null);
  const [cropRect, setCropRect] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });
  const [imgScale, setImgScale] = useState(1);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      const container = containerRef.current;
      const maxW = container ? container.clientWidth : 360;
      const maxH = Math.min(window.innerHeight * 0.55, 460);
      const s = Math.min(1, maxW / img.naturalWidth, maxH / img.naturalHeight);
      const cw = Math.round(img.naturalWidth * s); const ch = Math.round(img.naturalHeight * s);
      setCanvasSize({ w: cw, h: ch }); setImgScale(s);
      setCropRect({ x: Math.round(cw * 0.1), y: Math.round(ch * 0.1), w: Math.round(cw * 0.8), h: Math.round(ch * 0.8) });
      setImageLoaded(true);
    };
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!imageLoaded || !cropRect || !canvasRef.current) return;
    const canvas = canvasRef.current; const ctx = canvas.getContext("2d");
    const { w: cw, h: ch } = canvasSize;
    ctx.clearRect(0, 0, cw, ch); ctx.drawImage(imgRef.current, 0, 0, cw, ch);
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(0, 0, cw, cropRect.y); ctx.fillRect(0, cropRect.y + cropRect.h, cw, ch - cropRect.y - cropRect.h);
    ctx.fillRect(0, cropRect.y, cropRect.x, cropRect.h); ctx.fillRect(cropRect.x + cropRect.w, cropRect.y, cw - cropRect.x - cropRect.w, cropRect.h);
    ctx.strokeStyle = "#006D77"; ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeRect(cropRect.x, cropRect.y, cropRect.w, cropRect.h);
    ctx.strokeStyle = "rgba(0,109,119,0.3)"; ctx.lineWidth = 1;
    for (let i = 1; i <= 2; i++) {
      const gx = cropRect.x + (cropRect.w / 3) * i; const gy = cropRect.y + (cropRect.h / 3) * i;
      ctx.beginPath(); ctx.moveTo(gx, cropRect.y); ctx.lineTo(gx, cropRect.y + cropRect.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cropRect.x, gy); ctx.lineTo(cropRect.x + cropRect.w, gy); ctx.stroke();
    }
    ctx.fillStyle = "#006D77";
    [[cropRect.x, cropRect.y], [cropRect.x + cropRect.w, cropRect.y],
     [cropRect.x, cropRect.y + cropRect.h], [cropRect.x + cropRect.w, cropRect.y + cropRect.h]]
      .forEach(([cx, cy]) => ctx.fillRect(cx - 5, cy - 5, 10, 10));
  }, [imageLoaded, cropRect, canvasSize]);

  const getPos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (cx - rect.left) * (canvas.width / rect.width), y: (cy - rect.top) * (canvas.height / rect.height) };
  };
  const getZone = (pos, r) => {
    const hs = 18;
    const corners = { nw: [r.x, r.y], ne: [r.x + r.w, r.y], sw: [r.x, r.y + r.h], se: [r.x + r.w, r.y + r.h] };
    for (const [k, [cx, cy]] of Object.entries(corners)) if (Math.abs(pos.x - cx) <= hs && Math.abs(pos.y - cy) <= hs) return k;
    if (pos.x >= r.x && pos.x <= r.x + r.w && pos.y >= r.y && pos.y <= r.y + r.h) return "move";
    return null;
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const onDown = useCallback((e) => {
    if (!cropRect || !canvasRef.current) return; e.preventDefault();
    const pos = getPos(e, canvasRef.current); const zone = getZone(pos, cropRect);
    if (!zone) return;
    isDragging.current = true; dragMode.current = zone; dragStart.current = pos; cropAtStart.current = { ...cropRect };
  }, [cropRect]);
  const onMove = useCallback((e) => {
    if (!isDragging.current || !cropAtStart.current) return; e.preventDefault();
    const pos = getPos(e, canvasRef.current); const dx = pos.x - dragStart.current.x, dy = pos.y - dragStart.current.y;
    const { w: cw, h: ch } = canvasSize; const b = cropAtStart.current, MIN = 40, m = dragMode.current;
    setCropRect(() => {
      let { x, y, w, h } = b;
      if (m === "move") { x = clamp(b.x + dx, 0, cw - b.w); y = clamp(b.y + dy, 0, ch - b.h); }
      else if (m === "se") { w = clamp(b.w + dx, MIN, cw - b.x); h = clamp(b.h + dy, MIN, ch - b.y); }
      else if (m === "sw") { const nx = clamp(b.x + dx, 0, b.x + b.w - MIN); w = b.x + b.w - nx; x = nx; h = clamp(b.h + dy, MIN, ch - b.y); }
      else if (m === "ne") { w = clamp(b.w + dx, MIN, cw - b.x); const ny = clamp(b.y + dy, 0, b.y + b.h - MIN); h = b.y + b.h - ny; y = ny; }
      else if (m === "nw") { const nx = clamp(b.x + dx, 0, b.x + b.w - MIN); const ny = clamp(b.y + dy, 0, b.y + b.h - MIN); w = b.x + b.w - nx; x = nx; h = b.y + b.h - ny; y = ny; }
      return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
    });
  }, [canvasSize]);
  const onUp = useCallback((e) => { e.preventDefault(); isDragging.current = false; }, []);
  const handleConfirm = () => {
    if (!cropRect || !imgScale) { onConfirm(null); return; }
    onConfirm({ x: Math.round(cropRect.x / imgScale), y: Math.round(cropRect.y / imgScale), width: Math.round(cropRect.w / imgScale), height: Math.round(cropRect.h / imgScale) });
  };

  return (
    <div style={overlayBg}>
      <div style={{ width: "100%", maxWidth: 520 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div>
            <p style={{ fontSize: 13, fontWeight: 700, color: "var(--white)", display: "flex", alignItems: "center", gap: 8 }}>
              <Icon n="crop" size={14} style={{ color: "var(--mint)" }} /> Crop Nutrition Label
            </p>
            <p style={{ fontSize: 11, color: "rgba(253,252,249,0.5)", marginTop: 2 }}>Drag corners to resize · Drag inside to move</p>
          </div>
          <button onClick={onCancel} style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(253,252,249,0.5)" }}><Icon n="close" size={18} /></button>
        </div>
        <div ref={containerRef} style={{ width: "100%", borderRadius: 14, overflow: "hidden", border: "1px solid var(--border2)", touchAction: "none", userSelect: "none" }}>
          {imageLoaded
            ? <canvas ref={canvasRef} width={canvasSize.w} height={canvasSize.h} style={{ display: "block", width: "100%", touchAction: "none", cursor: "crosshair" }}
                onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
                onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp} />
            : <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 240 }}><Spin size={24} /></div>
          }
        </div>
        {cropRect && <p style={{ fontSize: 10, color: "rgba(253,252,249,0.4)", textAlign: "center", marginTop: 8 }}>REGION: {Math.round(cropRect.w / imgScale)}×{Math.round(cropRect.h / imgScale)}px</p>}
        <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
          <button onClick={onCancel} style={{ flex: 1, padding: "11px", background: "rgba(253,252,249,0.08)", border: "1px solid rgba(253,252,249,0.2)", borderRadius: 10, fontSize: 13, fontWeight: 600, color: "rgba(253,252,249,0.7)", cursor: "pointer" }}>Cancel</button>
          <button onClick={() => onConfirm(null)} style={{ flex: 1, padding: "11px", background: "rgba(253,252,249,0.08)", border: "1px solid rgba(253,252,249,0.2)", borderRadius: 10, fontSize: 13, fontWeight: 600, color: "rgba(253,252,249,0.7)", cursor: "pointer" }}>Full image</button>
          <button onClick={handleConfirm} style={{ flex: 1, padding: "11px", background: "var(--teal)", border: "none", borderRadius: 10, fontSize: 13, fontWeight: 700, color: "white", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
            <Icon n="check" size={14} /> Confirm
          </button>
        </div>
      </div>
    </div>
  );
}
