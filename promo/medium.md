# I Built an App That Reads Nutrition Labels From a Photo

## Scanning a label beats typing it every time

Every macro tracker has the same friction. You eat something, then you hunt through a database, or you type numbers off the back of a packet by hand. For anything not in the app's catalog, you are guessing.

So I built **NutriScan**. You take a photo of the nutrition label, and it reads the calories, protein, carbs, fat and fibre off it. You log it in one tap, and it counts toward your daily goals. No barcode lookup, no manual entry.

**Try it right now (installable on your phone):**
**https://nutritional-tracker-delta.vercel.app**

## What it does

- **Scan.** Photograph any label, get all the macros instantly. Up to 10 at once.
- **Track.** Daily totals versus your personal goals, with progress bars and a calendar for past days.
- **Library and templates.** Save foods into folders, and build a "Breakfast" template that logs a whole meal in one tap.
- **Assistant.** A chat that already knows today's log, your remaining macros, and your 7-day averages.
- **Notifications.** Meal reminders at times you pick, plus goal and scan-limit alerts, delivered as real push notifications.
- **Install it.** Add to your home screen on iOS or Android and it runs like a native app, offline, in light or dark mode.

## How it's built

The interesting engineering is under the hood:

- **Vision.** The label reader runs on Google Gemini, with a fallback model when the primary is unavailable.
- **Security.** Every table has Postgres row-level security, so a user's data is isolated by the database itself, not just by application code. The app connects with a least-privilege role that cannot bypass those policies.
- **Auth.** Google sign-in and emailed one-time codes, with tokens verified server-side against a JWKS endpoint.
- **Cost control.** A daily scan limit enforced before any expensive call, rate limiting on chat and public endpoints, an upload size cap, and a serverless database that sleeps when idle.
- **PWA.** Offline-capable, with a versioned update system that shows users exactly what each update contains before they apply it.

Stack: React 19 and Vite on the front, FastAPI and Neon serverless Postgres on the back, Supabase for auth, deployed across Vercel and Render.

## Try it and tell me what breaks

It is free, and it is genuinely useful if you track your macros. I would love feedback, especially the ugly bits.

- **Use it:** https://nutritional-tracker-delta.vercel.app
- **Code (open source):** https://github.com/MrTig-afk/NutritionalTracker
- **Connect with me:** https://www.linkedin.com/in/kaushikn2002/
