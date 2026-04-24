const MAX_IMAGE_PX = 1024;
const JPEG_QUALITY = 0.85;

export const NUTRIENT_META = {
  calories:      { label: "Calories",       unit: "kcal", color: "var(--orange)"  },
  fat:           { label: "Total Fat",       unit: "g",    color: "var(--brown)"   },
  saturated_fat: { label: "Saturated Fat",   unit: "g",    color: "var(--danger)"  },
  carbohydrates: { label: "Carbohydrates",   unit: "g",    color: "var(--purple)"  },
  sugars:        { label: "of which Sugars", unit: "g",    color: "var(--purple)"  },
  fibre:         { label: "Dietary Fibre",   unit: "g",    color: "var(--mint-dk)" },
  protein:       { label: "Protein",         unit: "g",    color: "var(--teal)"    },
  sodium:        { label: "Sodium",          unit: "g",    color: "var(--muted)"   },
};

export function getFallbackMeta(key) {
  return { label: key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()), unit: "", color: "var(--muted)" };
}

export function parseNumeric(value) {
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const match = value.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : null;
  }
  return null;
}

export function extractServingGrams(size) {
  if (!size) return null;
  const gMatch = size.match(/(\d+(\.\d+)?)\s*g/i);
  if (gMatch) return parseFloat(gMatch[1]);
  const numMatch = size.match(/(\d+(\.\d+)?)/);
  return numMatch ? parseFloat(numMatch[1]) : null;
}

export function normalizeCalories(value) {
  const num = parseNumeric(value);
  if (num === null) return value;
  if (typeof value === "string" && /kj|kilojoule/i.test(value)) return Math.round(num / 4.184);
  return value;
}

export function normalizeNutritionData(data) {
  if (!data || typeof data !== "object") return data;
  const result = { ...data };
  if (result.calories !== undefined) result.calories = normalizeCalories(result.calories);
  return result;
}

export function normalizeResult(result) {
  if (!result) return result;
  return {
    ...result,
    per_serving: result.per_serving ? normalizeNutritionData(result.per_serving) : result.per_serving,
    per_100g:    result.per_100g    ? normalizeNutritionData(result.per_100g)    : result.per_100g,
  };
}

export function resolveNutrition(nutrition) {
  if (!nutrition) return {};
  if (!nutrition.per_serving && !nutrition.per_100g) return nutrition;
  if (nutrition.per_serving && Object.keys(nutrition.per_serving).length > 0) return nutrition.per_serving;
  if (nutrition.per_100g    && Object.keys(nutrition.per_100g).length    > 0) return nutrition.per_100g;
  return nutrition;
}

export function formatDisplayDate(isoDate) {
  const [year, month, day] = isoDate.split("-").map(Number);
  const d = new Date(year, month - 1, day);
  const dayName = d.toLocaleDateString("en-US", { weekday: "long" });
  return `${String(day).padStart(2, "0")}-${String(month).padStart(2, "0")}-${year} (${dayName})`;
}

export function addDays(isoDate, n) {
  const [year, month, day] = isoDate.split("-").map(Number);
  const d = new Date(year, month - 1, day + n);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${dd}`;
}

export async function applyPipelineToFile(file, cropData) {
  return new Promise((resolve) => {
    const img = new window.Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const srcX = cropData ? cropData.x : 0; const srcY = cropData ? cropData.y : 0;
      const srcW = cropData ? cropData.width : img.naturalWidth;
      const srcH = cropData ? cropData.height : img.naturalHeight;
      const scale = Math.min(1, MAX_IMAGE_PX / Math.max(srcW, srcH));
      const tW = Math.max(1, Math.round(srcW * scale)); const tH = Math.max(1, Math.round(srcH * scale));
      const canvas = document.createElement("canvas"); canvas.width = tW; canvas.height = tH;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, srcX, srcY, srcW, srcH, 0, 0, tW, tH);
      const id = ctx.getImageData(0, 0, tW, tH); const d = id.data;
      for (let i = 0; i < d.length; i += 4) { const g = 0.299*d[i]+0.587*d[i+1]+0.114*d[i+2]; d[i]=d[i+1]=d[i+2]=g; }
      ctx.putImageData(id, 0, 0);
      canvas.toBlob((blob) => {
        if (!blob) { resolve(file); return; }
        resolve(new File([blob], file.name.replace(/\.[^.]+$/, ".jpg"), { type: "image/jpeg", lastModified: Date.now() }));
      }, "image/jpeg", JPEG_QUALITY);
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

export function createPreviewUrl(file) { return URL.createObjectURL(file); }
