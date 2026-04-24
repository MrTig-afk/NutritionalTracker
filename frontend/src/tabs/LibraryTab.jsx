import React, { useState, useEffect, useCallback } from "react";
import { apiFetch } from "../lib/api";
import { parseNumeric } from "../lib/nutrition";
import { card, inputStyle, primaryBtn } from "../styles";
import { Icon, Spin } from "../components/Icon";

export default function LibraryTab({ onAddToLog }) {
  const [folders, setFolders] = useState([]);
  const [openFolder, setOpenFolder] = useState(null);
  const [folderData, setFolderData] = useState({});
  const [newFolderName, setNewFolderName] = useState("");
  const [creating, setCreating] = useState(false);
  const [loading, setLoading] = useState(true);
  const [deletingItem, setDeletingItem] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");

  const loadFolders = useCallback(async () => {
    setLoading(true); setFolderData({});
    try {
      const fl = await apiFetch("/folders");
      setFolders(fl);
      fl.forEach(f => {
        apiFetch(`/folders/${f.folder_id}`)
          .then(data => setFolderData(prev => ({ ...prev, [f.folder_id]: data })))
          .catch(() => {});
      });
    }
    catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { loadFolders(); }, [loadFolders]);

  const openFolderById = async (id) => {
    if (openFolder === id) { setOpenFolder(null); return; }
    setOpenFolder(id);
    try { const data = await apiFetch(`/folders/${id}`); setFolderData(prev => ({ ...prev, [id]: data })); }
    catch (e) { console.error(e); }
  };

  const createFolder = async () => {
    if (!newFolderName.trim()) return; setCreating(true);
    try { const f = await apiFetch("/folders", { method: "POST", body: JSON.stringify({ name: newFolderName.trim() }) }); setFolders(prev => [f, ...prev]); setNewFolderName(""); }
    catch (e) { console.error(e); }
    finally { setCreating(false); }
  };

  const handleFolderKeyDown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); createFolder(); return; }
    if (e.ctrlKey && e.key === "Backspace") { e.preventDefault(); setNewFolderName(prev => prev.replace(/\s*\S+\s*$/, "")); return; }
    if (e.ctrlKey && e.key === "Delete") { e.preventDefault(); const pos = e.target.selectionStart; setNewFolderName(newFolderName.slice(0, pos) + newFolderName.slice(pos).replace(/^\s*\S+/, "")); }
  };

  const deleteItem = async (folderId, itemId) => {
    setDeletingItem(itemId);
    try { await apiFetch(`/folders/${folderId}/items/${itemId}`, { method: "DELETE" }); setFolderData(prev => ({ ...prev, [folderId]: { ...prev[folderId], items: prev[folderId].items.filter(i => i.item_id !== itemId) } })); }
    catch (e) { console.error(e); }
    finally { setDeletingItem(null); }
  };

  const deleteFolder = async (folderId) => {
    try { await apiFetch(`/folders/${folderId}`, { method: "DELETE" }); setFolders(prev => prev.filter(f => f.folder_id !== folderId)); if (openFolder === folderId) setOpenFolder(null); }
    catch (e) { console.error(e); }
  };

  if (loading) return <div style={{ display: "flex", justifyContent: "center", padding: "60px 0" }}><Spin size={24} /></div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", gap: 8 }}>
        <input value={newFolderName} onChange={e => setNewFolderName(e.target.value)} placeholder="New folder name..." onKeyDown={handleFolderKeyDown}
          style={{ ...inputStyle, flex: 1 }} />
        <button onClick={createFolder} disabled={creating || !newFolderName.trim()}
          style={{ ...primaryBtn, width: "auto", padding: "9px 16px", fontSize: 13, opacity: (creating || !newFolderName.trim()) ? 0.45 : 1 }}>
          {creating ? <Spin size={13} /> : <Icon n="create_new_folder" size={13} />} Create
        </button>
      </div>

      <input
        value={searchQuery}
        onChange={e => setSearchQuery(e.target.value)}
        placeholder="Search items across all folders..."
        style={inputStyle}
      />

      {searchQuery.trim() ? (
        (() => {
          const q = searchQuery.toLowerCase();
          const hits = folders.flatMap(folder =>
            (folderData[folder.folder_id]?.items || [])
              .filter(item => item.name.toLowerCase().includes(q))
              .map(item => ({ ...item, folderName: folder.name, folderId: folder.folder_id }))
          );
          return hits.length === 0
            ? <div style={{ padding: "32px 16px", textAlign: "center", color: "var(--muted)", fontSize: 14, fontStyle: "italic", border: "2px dashed var(--border)", borderRadius: 16 }}>No items match "{searchQuery}"</div>
            : <div style={card}>{hits.map((item, idx) => {
                const nutrition = item.nutrition?.per_serving ?? item.nutrition ?? {};
                const cal  = parseNumeric(nutrition.calories)      || 0;
                const prot = parseNumeric(nutrition.protein)       || 0;
                const carb = parseNumeric(nutrition.carbohydrates) || 0;
                const fat  = parseNumeric(nutrition.fat)           || 0;
                const imageUrl = item.nutrition?.processed_url || item.nutrition?.raw_url || null;
                return (
                  <div key={item.item_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderTop: idx === 0 ? "none" : "1px solid var(--off2)" }}>
                    {imageUrl
                      ? <div style={{ width: 44, height: 44, borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)", flexShrink: 0 }}><img src={imageUrl} alt={item.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} onError={e => { e.target.style.display = "none"; }} /></div>
                      : <div style={{ width: 44, height: 44, borderRadius: 10, background: "var(--teal-lt)", border: "1px solid var(--border)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="storage" size={14} style={{ color: "var(--teal)" }} /></div>
                    }
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</div>
                      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{item.folderName} · {cal}kcal · P {prot}g · C {carb}g · F {fat}g</div>
                    </div>
                    <button onClick={() => onAddToLog({ ...item })} style={{ padding: "5px 10px", background: "var(--mint)", border: "none", borderRadius: 8, fontSize: 11, fontWeight: 700, color: "var(--mint-dk)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
                      <Icon n="add" size={11} /> Log
                    </button>
                  </div>
                );
              })}</div>;
        })()
      ) : folders.length === 0 ? (
        <div style={{ padding: "48px 16px", textAlign: "center", color: "var(--muted)", fontSize: 14, fontStyle: "italic", border: "2px dashed var(--border)", borderRadius: 16 }}>No folders yet. Create one above.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {folders.map(folder => (
            <div key={folder.folder_id} style={card}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", cursor: "pointer" }} onClick={() => openFolderById(folder.folder_id)}>
                {openFolder === folder.folder_id
                  ? <Icon n="folder_open" size={18} style={{ color: "var(--teal)" }} />
                  : <Icon n="folder" size={18} style={{ color: "var(--muted)" }} />}
                <span style={{ flex: 1, fontSize: 14, fontWeight: 600, color: "var(--text)" }}>{folder.name}</span>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>{folderData[folder.folder_id]?.items?.length ?? ""} items</span>
                <button onClick={e => { e.stopPropagation(); deleteFolder(folder.folder_id); }} style={{ background: "none", border: "none", cursor: "pointer", marginLeft: 4 }}>
                  <Icon n="delete" size={13} style={{ color: "var(--muted)" }} />
                </button>
                {openFolder === folder.folder_id ? <Icon n="expand_less" size={14} style={{ color: "var(--muted)" }} /> : <Icon n="expand_more" size={14} style={{ color: "var(--muted)" }} />}
              </div>

              {openFolder === folder.folder_id && folderData[folder.folder_id] && (
                <div style={{ borderTop: "1px solid var(--off2)" }}>
                  {folderData[folder.folder_id].items.length === 0 ? (
                    <p style={{ padding: "16px", textAlign: "center", color: "var(--muted)", fontSize: 13, fontStyle: "italic" }}>No items in this folder.</p>
                  ) : (
                    folderData[folder.folder_id].items.map((item, idx) => {
                      const nutrition = item.nutrition?.per_serving ?? item.nutrition ?? {};
                      const cal  = parseNumeric(nutrition.calories)      || 0;
                      const prot = parseNumeric(nutrition.protein)       || 0;
                      const carb = parseNumeric(nutrition.carbohydrates) || 0;
                      const fat  = parseNumeric(nutrition.fat)           || 0;
                      const imageUrl = item.nutrition?.processed_url || item.nutrition?.raw_url || item.processed_url || item.raw_url || null;
                      return (
                        <div key={item.item_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px 10px 20px", borderTop: idx === 0 ? "none" : "1px solid var(--off2)" }}>
                          {imageUrl
                            ? <div style={{ width: 44, height: 44, borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)", flexShrink: 0 }}><img src={imageUrl} alt={item.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} onError={e => { e.target.style.display = "none"; }} /></div>
                            : <div style={{ width: 44, height: 44, borderRadius: 10, background: "var(--teal-lt)", border: "1px solid var(--border)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="storage" size={14} style={{ color: "var(--teal)" }} /></div>
                          }
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</div>
                            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{cal}kcal · P {prot}g · C {carb}g · F {fat}g</div>
                          </div>
                          <button onClick={() => onAddToLog({ ...item })} style={{ padding: "5px 10px", background: "var(--mint)", border: "none", borderRadius: 8, fontSize: 11, fontWeight: 700, color: "var(--mint-dk)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
                            <Icon n="add" size={11} /> Log
                          </button>
                          <button onClick={() => deleteItem(folder.folder_id, item.item_id)} disabled={deletingItem === item.item_id} style={{ background: "none", border: "none", cursor: "pointer" }}>
                            {deletingItem === item.item_id ? <Spin size={13} color="var(--muted)" /> : <Icon n="delete" size={13} style={{ color: "var(--muted)" }} />}
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
