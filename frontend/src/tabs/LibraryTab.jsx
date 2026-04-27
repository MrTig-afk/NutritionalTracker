import React, { useState, useEffect, useCallback } from "react";
import { apiFetch } from "../lib/api";
import { parseNumeric } from "../lib/nutrition";
import { card, inputStyle, primaryBtn } from "../styles";
import { Icon, Spin } from "../components/Icon";

export default function LibraryTab({ onAddToLog }) {
  // ── Folders ──────────────────────────────────────────────────────────────
  const [folders, setFolders]         = useState([]);
  const [openFolder, setOpenFolder]   = useState(null);
  const [folderData, setFolderData]   = useState({});
  const [newFolderName, setNewFolderName] = useState("");
  const [creating, setCreating]       = useState(false);
  const [loading, setLoading]         = useState(true);
  const [deletingItem, setDeletingItem] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");

  // ── Meal Templates ───────────────────────────────────────────────────────
  const [templates, setTemplates]     = useState([]);
  const [templateData, setTemplateData] = useState({});
  const [openTemplate, setOpenTemplate] = useState(null);
  const [newTmplName, setNewTmplName] = useState("");
  const [creatingTmpl, setCreatingTmpl] = useState(false);
  const [loggingTmpl, setLoggingTmpl] = useState(null);
  const [deletingTmplItem, setDeletingTmplItem] = useState(null);
  const [itemPickerFor, setItemPickerFor] = useState(null);
  const [itemPickerQuery, setItemPickerQuery] = useState("");


  // ── Load data ─────────────────────────────────────────────────────────────
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
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  const loadTemplates = useCallback(async () => {
    try {
      const tl = await apiFetch("/meal-templates");
      setTemplates(tl);
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { loadFolders(); loadTemplates(); }, [loadFolders, loadTemplates]);


  // ── Folders ───────────────────────────────────────────────────────────────
  const openFolderById = async (id) => {
    if (openFolder === id) { setOpenFolder(null); return; }
    setOpenFolder(id);
    try { const data = await apiFetch(`/folders/${id}`); setFolderData(prev => ({ ...prev, [id]: data })); }
    catch (e) { console.error(e); }
  };

  const createFolder = async () => {
    if (!newFolderName.trim()) return; setCreating(true);
    try {
      const f = await apiFetch("/folders", { method: "POST", body: JSON.stringify({ name: newFolderName.trim() }) });
      setFolders(prev => [f, ...prev]); setNewFolderName("");
    } catch (e) { console.error(e); }
    finally { setCreating(false); }
  };

  const handleFolderKeyDown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); createFolder(); return; }
    if (e.ctrlKey && e.key === "Backspace") { e.preventDefault(); setNewFolderName(prev => prev.replace(/\s*\S+\s*$/, "")); return; }
    if (e.ctrlKey && e.key === "Delete") { e.preventDefault(); const pos = e.target.selectionStart; setNewFolderName(newFolderName.slice(0, pos) + newFolderName.slice(pos).replace(/^\s*\S+/, "")); }
  };

  const deleteItem = async (folderId, itemId) => {
    setDeletingItem(itemId);
    try {
      await apiFetch(`/folders/${folderId}/items/${itemId}`, { method: "DELETE" });
      setFolderData(prev => ({ ...prev, [folderId]: { ...prev[folderId], items: prev[folderId].items.filter(i => i.item_id !== itemId) } }));
    } catch (e) { console.error(e); }
    finally { setDeletingItem(null); }
  };

  const deleteFolder = async (folderId) => {
    try {
      await apiFetch(`/folders/${folderId}`, { method: "DELETE" });
      setFolders(prev => prev.filter(f => f.folder_id !== folderId));
      if (openFolder === folderId) setOpenFolder(null);
    } catch (e) { console.error(e); }
  };

  // ── Meal Templates ────────────────────────────────────────────────────────
  const createTemplate = async () => {
    if (!newTmplName.trim()) return; setCreatingTmpl(true);
    try {
      const t = await apiFetch("/meal-templates", { method: "POST", body: JSON.stringify({ name: newTmplName.trim() }) });
      setTemplates(prev => [t, ...prev]);
      setNewTmplName("");
      setOpenTemplate(t.template_id);
      setTemplateData(prev => ({ ...prev, [t.template_id]: { ...t, items: [] } }));
    } catch (e) { console.error(e); }
    finally { setCreatingTmpl(false); }
  };

  const openTemplateById = async (id) => {
    if (openTemplate === id) { setOpenTemplate(null); return; }
    setOpenTemplate(id);
    if (!templateData[id]) {
      try { const data = await apiFetch(`/meal-templates/${id}`); setTemplateData(prev => ({ ...prev, [id]: data })); }
      catch (e) { console.error(e); }
    }
  };

  const deleteTemplate = async (id) => {
    try {
      await apiFetch(`/meal-templates/${id}`, { method: "DELETE" });
      setTemplates(prev => prev.filter(t => t.template_id !== id));
      if (openTemplate === id) setOpenTemplate(null);
    } catch (e) { console.error(e); }
  };

  const deleteTmplItem = async (templateId, itemId) => {
    setDeletingTmplItem(itemId);
    try {
      await apiFetch(`/meal-templates/${templateId}/items/${itemId}`, { method: "DELETE" });
      setTemplateData(prev => ({
        ...prev,
        [templateId]: { ...prev[templateId], items: prev[templateId].items.filter(i => i.item_id !== itemId) },
      }));
      setTemplates(prev => prev.map(t => t.template_id === templateId ? { ...t, item_count: t.item_count - 1 } : t));
    } catch (e) { console.error(e); }
    finally { setDeletingTmplItem(null); }
  };

  const addLibraryItemToTemplate = async (templateId, libraryItem) => {
    const nutrition = libraryItem.nutrition || {};
    try {
      const item = await apiFetch(`/meal-templates/${templateId}/items`, {
        method: "POST",
        body: JSON.stringify({ name: libraryItem.name, nutrition, servings: 1 }),
      });
      setTemplateData(prev => ({
        ...prev,
        [templateId]: { ...prev[templateId], items: [...(prev[templateId]?.items || []), { ...item, nutrition }] },
      }));
      setTemplates(prev => prev.map(t => t.template_id === templateId ? { ...t, item_count: t.item_count + 1 } : t));
      setItemPickerFor(null);
      setItemPickerQuery("");
    } catch (e) { console.error(e); }
  };

  const logTemplate = async (templateId) => {
    setLoggingTmpl(templateId);
    try {
      const res = await apiFetch(`/meal-templates/${templateId}/log`, { method: "POST" });
      if (res.logged > 0) onAddToLog(null);
    } catch (e) { console.error(e); }
    finally { setLoggingTmpl(null); }
  };

  // ── Helpers ───────────────────────────────────────────────────────────────
  const allLibraryItems = folders.flatMap(f =>
    (folderData[f.folder_id]?.items || []).map(i => ({ ...i, folderName: f.name }))
  );

  const macroLine = (nutrition) => {
    const n = nutrition?.per_serving ?? nutrition ?? {};
    const cal  = parseNumeric(n.calories)      || 0;
    const prot = parseNumeric(n.protein)       || 0;
    const carb = parseNumeric(n.carbohydrates) || 0;
    const fat  = parseNumeric(n.fat)           || 0;
    return `${cal}kcal · P ${prot}g · C ${carb}g · F ${fat}g`;
  };

  if (loading) return <div style={{ display: "flex", justifyContent: "center", padding: "60px 0" }}><Spin size={24} /></div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

      {/* ── Create folder ── */}
      <div style={{ display: "flex", gap: 8 }}>
        <input value={newFolderName} onChange={e => setNewFolderName(e.target.value)} placeholder="New folder name..."
          onKeyDown={handleFolderKeyDown} style={{ ...inputStyle, flex: 1 }} />
        <button onClick={createFolder} disabled={creating || !newFolderName.trim()}
          style={{ ...primaryBtn, width: "auto", padding: "9px 16px", fontSize: 13, opacity: (creating || !newFolderName.trim()) ? 0.45 : 1 }}>
          {creating ? <Spin size={13} /> : <Icon n="create_new_folder" size={13} />} Create
        </button>
      </div>

      {/* ── Search ── */}
      <input value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
        placeholder="Search items across all folders..." style={inputStyle} />

      {/* ── Search results ── */}
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
            : <div style={card}>{hits.map((item, idx) => (
                <div key={item.item_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderTop: idx === 0 ? "none" : "1px solid var(--off2)" }}>
                  {(item.nutrition?.processed_url || item.nutrition?.raw_url)
                    ? <div style={{ width: 44, height: 44, borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)", flexShrink: 0 }}><img src={item.nutrition.processed_url || item.nutrition.raw_url} alt={item.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} onError={e => { e.target.style.display = "none"; }} /></div>
                    : <div style={{ width: 44, height: 44, borderRadius: 10, background: "var(--teal-lt)", border: "1px solid var(--border)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="storage" size={14} style={{ color: "var(--teal)" }} /></div>
                  }
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</div>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{item.folderName} · {macroLine(item.nutrition)}</div>
                  </div>
                  <button onClick={() => onAddToLog({ ...item })} style={{ padding: "5px 10px", background: "var(--mint)", border: "none", borderRadius: 8, fontSize: 11, fontWeight: 700, color: "var(--mint-dk)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
                    <Icon n="add" size={11} /> Log
                  </button>
                </div>
              ))}</div>;
        })()
      ) : (
        <>
          {/* ── Folder list ── */}
          {folders.length === 0 ? (
            <div style={{ padding: "48px 16px", textAlign: "center", color: "var(--muted)", fontSize: 14, fontStyle: "italic", border: "2px dashed var(--border)", borderRadius: 16 }}>No folders yet. Create one above.</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {folders.map(folder => (
                <div key={folder.folder_id} style={card}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", cursor: "pointer" }} onClick={() => openFolderById(folder.folder_id)}>
                    {openFolder === folder.folder_id ? <Icon n="folder_open" size={18} style={{ color: "var(--teal)" }} /> : <Icon n="folder" size={18} style={{ color: "var(--muted)" }} />}
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
                          const imageUrl = item.nutrition?.processed_url || item.nutrition?.raw_url || item.processed_url || item.raw_url || null;
                          return (
                            <div key={item.item_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px 10px 20px", borderTop: idx === 0 ? "none" : "1px solid var(--off2)" }}>
                              {imageUrl
                                ? <div style={{ width: 44, height: 44, borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)", flexShrink: 0 }}><img src={imageUrl} alt={item.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} onError={e => { e.target.style.display = "none"; }} /></div>
                                : <div style={{ width: 44, height: 44, borderRadius: 10, background: "var(--teal-lt)", border: "1px solid var(--border)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="storage" size={14} style={{ color: "var(--teal)" }} /></div>
                              }
                              <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</div>
                                <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{macroLine(item.nutrition)}</div>
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

          {/* ── Meal Templates section ── */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
            <Icon n="restaurant_menu" size={16} style={{ color: "var(--teal)" }} />
            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)", flex: 1 }}>Meal Templates</span>
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <input value={newTmplName} onChange={e => setNewTmplName(e.target.value)} placeholder="New template name..."
              onKeyDown={e => e.key === "Enter" && createTemplate()} style={{ ...inputStyle, flex: 1 }} />
            <button onClick={createTemplate} disabled={creatingTmpl || !newTmplName.trim()}
              style={{ ...primaryBtn, width: "auto", padding: "9px 16px", fontSize: 13, opacity: (creatingTmpl || !newTmplName.trim()) ? 0.45 : 1 }}>
              {creatingTmpl ? <Spin size={13} /> : <Icon n="add" size={13} />} New
            </button>
          </div>

          {templates.length === 0 ? (
            <div style={{ padding: "32px 16px", textAlign: "center", color: "var(--muted)", fontSize: 14, fontStyle: "italic", border: "2px dashed var(--border)", borderRadius: 16 }}>
              No templates yet. Create one to log a set of foods in one tap.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {templates.map(tmpl => (
                <div key={tmpl.template_id} style={card}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px" }}>
                    <div style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: 8, flex: 1 }} onClick={() => openTemplateById(tmpl.template_id)}>
                      <Icon n="set_meal" size={18} style={{ color: openTemplate === tmpl.template_id ? "var(--teal)" : "var(--muted)" }} />
                      <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text)" }}>{tmpl.name}</span>
                      <span style={{ fontSize: 11, color: "var(--muted)" }}>{tmpl.item_count} items</span>
                    </div>
                    <button onClick={() => logTemplate(tmpl.template_id)} disabled={loggingTmpl === tmpl.template_id || tmpl.item_count === 0}
                      style={{ padding: "5px 10px", background: "var(--mint)", border: "none", borderRadius: 8, fontSize: 11, fontWeight: 700, color: "var(--mint-dk)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4, opacity: tmpl.item_count === 0 ? 0.4 : 1 }}>
                      {loggingTmpl === tmpl.template_id ? <Spin size={11} /> : <Icon n="playlist_add_check" size={11} />} Log Meal
                    </button>
                    <button onClick={e => { e.stopPropagation(); deleteTemplate(tmpl.template_id); }} style={{ background: "none", border: "none", cursor: "pointer", marginLeft: 2 }}>
                      <Icon n="delete" size={13} style={{ color: "var(--muted)" }} />
                    </button>
                    <div onClick={() => openTemplateById(tmpl.template_id)} style={{ cursor: "pointer" }}>
                      {openTemplate === tmpl.template_id ? <Icon n="expand_less" size={14} style={{ color: "var(--muted)" }} /> : <Icon n="expand_more" size={14} style={{ color: "var(--muted)" }} />}
                    </div>
                  </div>

                  {openTemplate === tmpl.template_id && (
                    <div style={{ borderTop: "1px solid var(--off2)" }}>
                      {templateData[tmpl.template_id]?.items?.length === 0 && (
                        <p style={{ padding: "12px 16px", textAlign: "center", color: "var(--muted)", fontSize: 13, fontStyle: "italic" }}>No items yet.</p>
                      )}
                      {(templateData[tmpl.template_id]?.items || []).map((item, idx) => (
                        <div key={item.item_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px 10px 20px", borderTop: idx === 0 ? "none" : "1px solid var(--off2)" }}>
                          <div style={{ width: 36, height: 36, borderRadius: 8, background: "var(--teal-lt)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <Icon n="restaurant" size={13} style={{ color: "var(--teal)" }} />
                          </div>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</div>
                            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 1 }}>×{item.servings} · {macroLine(item.nutrition)}</div>
                          </div>
                          <button onClick={() => deleteTmplItem(tmpl.template_id, item.item_id)} disabled={deletingTmplItem === item.item_id} style={{ background: "none", border: "none", cursor: "pointer" }}>
                            {deletingTmplItem === item.item_id ? <Spin size={13} color="var(--muted)" /> : <Icon n="delete" size={13} style={{ color: "var(--muted)" }} />}
                          </button>
                        </div>
                      ))}

                      {/* Add from library */}
                      {itemPickerFor === tmpl.template_id ? (
                        <div style={{ padding: "10px 16px", borderTop: "1px solid var(--off2)" }}>
                          <input autoFocus value={itemPickerQuery} onChange={e => setItemPickerQuery(e.target.value)}
                            placeholder="Search library items..." style={{ ...inputStyle, marginBottom: 8 }} />
                          {(() => {
                            const q = itemPickerQuery.toLowerCase();
                            const hits = allLibraryItems.filter(i => !q || i.name.toLowerCase().includes(q)).slice(0, 12);
                            return hits.length === 0
                              ? <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: "8px 0" }}>No matches</div>
                              : hits.map(i => (
                                  <div key={i.item_id} onClick={() => addLibraryItemToTemplate(tmpl.template_id, i)}
                                    style={{ padding: "8px 10px", borderRadius: 8, cursor: "pointer", display: "flex", gap: 8, alignItems: "center" }}
                                    onMouseEnter={e => e.currentTarget.style.background = "var(--off2)"}
                                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{i.name}</div>
                                      <div style={{ fontSize: 11, color: "var(--muted)" }}>{i.folderName} · {macroLine(i.nutrition)}</div>
                                    </div>
                                    <Icon n="add_circle" size={16} style={{ color: "var(--teal)", flexShrink: 0 }} />
                                  </div>
                                ));
                          })()}
                          <button onClick={() => { setItemPickerFor(null); setItemPickerQuery(""); }}
                            style={{ marginTop: 6, background: "none", border: "none", color: "var(--muted)", fontSize: 12, cursor: "pointer" }}>
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <div style={{ padding: "10px 16px", borderTop: "1px solid var(--off2)" }}>
                          <button onClick={() => { setItemPickerFor(tmpl.template_id); setItemPickerQuery(""); }}
                            style={{ background: "none", border: "1px dashed var(--border)", borderRadius: 8, padding: "6px 14px", fontSize: 12, color: "var(--muted)", cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>
                            <Icon n="add" size={13} /> Add from library
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

        </>
      )}
    </div>
  );
}
