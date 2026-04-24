import React, { useState, useEffect } from "react";
import { apiFetch } from "../lib/api";
import { overlayBg, modalBox, modalHeader, modalTitle, inputStyle, labelStyle, primaryBtn, ghostBtn } from "../styles";
import { Icon, Spin } from "./Icon";

export default function SaveToFolderModal({ result, imageId, onClose, onSaved, initialName }) {
  const [folders, setFolders] = useState([]);
  const [newFolder, setNewFolder] = useState("");
  const [itemName, setItemName] = useState(initialName || "");
  const [selectedFolder, setSelectedFolder] = useState("");
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);

  useEffect(() => { apiFetch("/folders").then(setFolders).catch(() => {}); }, []);

  const createFolder = async () => {
    if (!newFolder.trim()) return;
    try {
      const f = await apiFetch("/folders", { method: "POST", body: JSON.stringify({ name: newFolder.trim() }) });
      setFolders(prev => [f, ...prev]); setSelectedFolder(f.folder_id); setNewFolder("");
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
  };
  const handleFolderKeyDown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); createFolder(); return; }
    if (e.ctrlKey && e.key === "Backspace") { e.preventDefault(); setNewFolder(prev => prev.replace(/\s*\S+\s*$/, "")); return; }
    if (e.ctrlKey && e.key === "Delete") { e.preventDefault(); const pos = e.target.selectionStart; setNewFolder(newFolder.slice(0, pos) + newFolder.slice(pos).replace(/^\s*\S+/, "")); }
  };
  const save = async () => {
    if (!selectedFolder || !itemName.trim()) return;
    setSaving(true);
    try {
      const nutrition = result.per_serving ? { per_serving: result.per_serving, per_100g: result.per_100g } : result;
      await apiFetch(`/folders/${selectedFolder}/items`, { method: "POST", body: JSON.stringify({ name: itemName.trim(), image_id: imageId || "", nutrition }) });
      setStatus({ type: "ok", msg: "Saved to folder!" });
      if (onSaved) onSaved(itemName.trim());
      setTimeout(onClose, 1200);
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
    finally { setSaving(false); }
  };

  return (
    <div style={overlayBg}>
      <div style={modalBox}>
        <div style={modalHeader}>
          <div style={modalTitle}><Icon n="bookmark_add" size={15} style={{ color: "var(--teal)" }} /> Save to Folder</div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted)" }}><Icon n="close" size={16} /></button>
        </div>
        {!initialName && (
          <div>
            <label style={labelStyle}>Item Name</label>
            <input value={itemName} onChange={e => setItemName(e.target.value)} placeholder="e.g. Greek Yogurt" style={inputStyle} />
          </div>
        )}
        <div>
          <label style={labelStyle}>Folder</label>
          <select value={selectedFolder} onChange={e => setSelectedFolder(e.target.value)} style={inputStyle}>
            <option value="">— select folder —</option>
            {folders.map(f => <option key={f.folder_id} value={f.folder_id}>{f.name}</option>)}
          </select>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <input value={newFolder} onChange={e => setNewFolder(e.target.value)} placeholder="New folder name..." onKeyDown={handleFolderKeyDown} style={{ ...inputStyle, flex: 1 }} />
          <button onClick={createFolder} style={{ ...ghostBtn, padding: "9px 12px" }}><Icon n="create_new_folder" size={14} /></button>
        </div>
        {status && <p style={{ fontSize: 12, color: status.type === "ok" ? "var(--mint-dk)" : "var(--danger)" }}>{status.msg}</p>}
        <button onClick={save} disabled={saving || !selectedFolder || !itemName.trim()} style={{ ...primaryBtn, opacity: (saving || !selectedFolder || !itemName.trim()) ? 0.45 : 1 }}>
          {saving ? <Spin size={14} /> : <Icon n="save" size={14} />}
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
    </div>
  );
}
