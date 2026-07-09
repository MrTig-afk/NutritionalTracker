/**
 * demoData.js — synthetic dataset for the PUBLIC DEMO build (VITE_DEMO=1).
 *
 * Everything here is invented: product names, macros, and the two-week diary
 * are generated in code, anchored to the visitor's local "today" so the
 * Tracker, calendar and Trends always look current. No real user records,
 * no real brands. Only apiDemo.js imports this module, and vite only aliases
 * apiDemo.js into demo builds, so none of this ships in the real app.
 */

// ── Products (invented brands) ───────────────────────────────────────────────

// Scan fixtures: these three match the bundled sample label images by file
// base name, so scanning a sample returns exactly what's printed on it.
export const SCAN_FIXTURES = {
  "morning-field-granola": {
    image_id: "demo-granola",
    per_serving: { size: "2/3 cup (55g)", calories: 231, fat: "7.8g", saturated_fat: "1.2g", carbohydrates: "36.5g", sugars: "11.8g", fibre: "4.2g", protein: "6.4g", sodium: "0.16g" },
    per_100g:    { calories: 420, fat: "14.2g", saturated_fat: "2.2g", carbohydrates: "66.4g", sugars: "21.5g", fibre: "7.6g", protein: "11.6g", sodium: "0.29g" },
  },
  "cloud-nine-yoghurt": {
    image_id: "demo-yoghurt",
    per_serving: { size: "1 tub (170g)", calories: 165, fat: "8.5g", saturated_fat: "5.4g", carbohydrates: "7.0g", sugars: "6.8g", fibre: "0g", protein: "15.3g", sodium: "0.085g" },
    per_100g:    { calories: 97, fat: "5.0g", saturated_fat: "3.2g", carbohydrates: "4.1g", sugars: "4.0g", fibre: "0g", protein: "9.0g", sodium: "0.05g" },
  },
  "peak-trail-bar": {
    image_id: "demo-bar",
    per_serving: { size: "1 bar (45g)", calories: 189, fat: "6.1g", saturated_fat: "2.4g", carbohydrates: "21.0g", sugars: "8.5g", fibre: "3.1g", protein: "12.2g", sodium: "0.12g" },
    per_100g:    { calories: 420, fat: "13.6g", saturated_fat: "5.3g", carbohydrates: "46.7g", sugars: "18.9g", fibre: "6.9g", protein: "27.1g", sodium: "0.27g" },
  },
};

export const FIXTURE_NAMES = {
  "morning-field-granola": "Morning Field Granola",
  "cloud-nine-yoghurt": "Cloud Nine Greek Yoghurt",
  "peak-trail-bar": "Peak Trail Protein Bar",
};

// Diary-only foods (never scanned, just logged).
const FOODS = {
  granola:   { name: "Morning Field Granola",     n: SCAN_FIXTURES["morning-field-granola"] },
  yoghurt:   { name: "Cloud Nine Greek Yoghurt",  n: SCAN_FIXTURES["cloud-nine-yoghurt"] },
  bar:       { name: "Peak Trail Protein Bar",    n: SCAN_FIXTURES["peak-trail-bar"] },
  oatmilk:   { name: "Sunrise Oat Milk",          n: { per_serving: { size: "1 cup (250ml)", calories: 120, fat: "3.0g", carbohydrates: "19.0g", sugars: "10.0g", fibre: "1.5g", protein: "2.5g" } } },
  toast:     { name: "Harvest Lane Sourdough Toast", n: { per_serving: { size: "2 slices (90g)", calories: 214, fat: "1.6g", carbohydrates: "41.0g", sugars: "2.4g", fibre: "3.8g", protein: "8.2g" } } },
  eggs:      { name: "Two Eggs, Scrambled",       n: { per_serving: { size: "2 eggs (120g)", calories: 182, fat: "13.4g", carbohydrates: "1.4g", sugars: "1.0g", fibre: "0g", protein: "14.0g" } } },
  wrap:      { name: "Ember Grill Chicken Wrap",  n: { per_serving: { size: "1 wrap (260g)", calories: 486, fat: "16.5g", carbohydrates: "48.0g", sugars: "5.5g", fibre: "5.2g", protein: "34.0g" } } },
  salad:     { name: "Green Basket Salad Bowl",   n: { per_serving: { size: "1 bowl (320g)", calories: 342, fat: "18.0g", carbohydrates: "27.0g", sugars: "8.0g", fibre: "7.5g", protein: "15.0g" } } },
  ramen:     { name: "Northside Ramen Kit",       n: { per_serving: { size: "1 bowl (450g)", calories: 610, fat: "21.0g", carbohydrates: "78.0g", sugars: "6.5g", fibre: "5.0g", protein: "27.0g" } } },
  stirfry:   { name: "Golden Wok Veggie Stir-fry", n: { per_serving: { size: "1 plate (380g)", calories: 452, fat: "14.0g", carbohydrates: "58.0g", sugars: "12.0g", fibre: "9.0g", protein: "20.0g" } } },
  pasta:     { name: "Casa Verde Pesto Pasta",    n: { per_serving: { size: "1 plate (340g)", calories: 574, fat: "22.0g", carbohydrates: "71.0g", sugars: "5.0g", fibre: "6.0g", protein: "19.0g" } } },
  cocoa:     { name: "Twilight Cocoa Bites",      n: { per_serving: { size: "4 pieces (30g)", calories: 158, fat: "9.5g", carbohydrates: "15.0g", sugars: "11.0g", fibre: "2.0g", protein: "2.4g" } } },
  cottage:   { name: "Blue Creek Cottage Cheese", n: { per_serving: { size: "1/2 cup (110g)", calories: 108, fat: "4.8g", carbohydrates: "3.9g", sugars: "3.5g", fibre: "0g", protein: "12.4g" } } },
  crackers:  { name: "Golden Crumb Seed Crackers", n: { per_serving: { size: "6 crackers (30g)", calories: 141, fat: "6.8g", carbohydrates: "15.5g", sugars: "1.2g", fibre: "3.0g", protein: "4.1g" } } },
  fruit:     { name: "Orchard Mix Fruit Cup",     n: { per_serving: { size: "1 cup (150g)", calories: 89, fat: "0.4g", carbohydrates: "21.0g", sugars: "17.0g", fibre: "3.2g", protein: "1.1g" } } },
};

// ── Date helpers (visitor-local, matching the app's own date formatting) ────

export function localISO(offsetDays = 0) {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// ── Deterministic PRNG so the diary is stable within a session ──────────────

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Build the two-week diary ─────────────────────────────────────────────────

let nextId = 1000;
export const newId = () => ++nextId;

export function buildDiary() {
  const rand = mulberry32(20260709);
  const entries = []; // { log_id, date, name, servings, nutrition }

  const add = (date, food, servings = 1, extra = {}) => {
    entries.push({ log_id: newId(), date, name: food.name, servings, nutrition: { ...food.n, ...extra } });
  };

  const LUNCHES  = [FOODS.wrap, FOODS.salad, FOODS.ramen, FOODS.cottage];
  const DINNERS  = [FOODS.stirfry, FOODS.pasta, FOODS.ramen, FOODS.salad];
  const SNACKS   = [FOODS.bar, FOODS.cocoa, FOODS.crackers, FOODS.fruit];

  for (let off = -13; off <= 0; off++) {
    const date = localISO(off);
    const today = off === 0;

    // Breakfast: most days the "Usual Breakfast" template (grouped block).
    if (rand() < 0.8) {
      const gid = `demo-breakfast-${date}`;
      const g = { _meal_group: gid, _meal_label: "Usual Breakfast" };
      add(date, FOODS.granola, 1, g);
      add(date, FOODS.yoghurt, 1, g);
      add(date, FOODS.oatmilk, 1, g);
    } else {
      add(date, FOODS.toast, 1);
      add(date, FOODS.eggs, 1);
    }

    // Lunch.
    add(date, LUNCHES[Math.floor(rand() * LUNCHES.length)], 1);

    // Today is caught mid-afternoon: breakfast + lunch + a snack, no dinner yet.
    add(date, SNACKS[Math.floor(rand() * SNACKS.length)], 1);
    if (!today) {
      add(date, DINNERS[Math.floor(rand() * DINNERS.length)], 1);
      if (rand() < 0.35) add(date, SNACKS[Math.floor(rand() * SNACKS.length)], 1);
    }
  }
  return entries;
}

// ── Initial app state ────────────────────────────────────────────────────────

export function buildInitialState() {
  const diary = buildDiary();

  const folders = [
    {
      folder_id: "demo-folder-breakfast", name: "Breakfast Staples",
      items: [
        { item_id: "demo-fi-1", name: FOODS.granola.name, nutrition: FOODS.granola.n },
        { item_id: "demo-fi-2", name: FOODS.yoghurt.name, nutrition: FOODS.yoghurt.n },
        { item_id: "demo-fi-3", name: FOODS.oatmilk.name, nutrition: FOODS.oatmilk.n },
      ],
    },
    {
      folder_id: "demo-folder-snacks", name: "Snacks",
      items: [
        { item_id: "demo-fi-4", name: FOODS.bar.name, nutrition: FOODS.bar.n },
        { item_id: "demo-fi-5", name: FOODS.cocoa.name, nutrition: FOODS.cocoa.n },
        { item_id: "demo-fi-6", name: FOODS.crackers.name, nutrition: FOODS.crackers.n },
      ],
    },
  ];

  const templates = [
    {
      template_id: "demo-tmpl-breakfast", name: "Usual Breakfast",
      items: [
        { item_id: "demo-ti-1", name: FOODS.granola.name, servings: 1, nutrition: FOODS.granola.n },
        { item_id: "demo-ti-2", name: FOODS.yoghurt.name, servings: 1, nutrition: FOODS.yoghurt.n },
        { item_id: "demo-ti-3", name: FOODS.oatmilk.name, servings: 1, nutrition: FOODS.oatmilk.n },
      ],
    },
  ];

  return {
    diary,
    folders,
    templates,
    goals: { calories: 2200, protein: 140, carbs: 260, fat: 70, fibre: 30 },
    usage: { used: 2, limit: 10 },
    prefs: {
      meal_morning: true,  meal_morning_time: "09:00",
      meal_afternoon: false, meal_afternoon_time: "13:00",
      meal_evening: true,  meal_evening_time: "19:30",
      weekly_summary: true,
    },
  };
}
