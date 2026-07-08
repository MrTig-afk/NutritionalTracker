# Reddit Post

Suggested subreddits: r/webdev, r/SideProject, r/reactjs, r/QuantifiedSelf, r/fitness (read rules first)

---

**Title:** I built a macro tracker that reads nutrition labels from a photo (no barcodes, no manual entry)

---

**Body:**

I got tired of typing macros off the back of packets, so I built an app that reads them for you. You take a photo of any nutrition label and it pulls the calories, protein, carbs, fat and fibre off it. Tap to log, and it counts toward your daily goals.

It is a PWA, so you can add it to your home screen and it runs like a native app, offline, with light and dark mode.

Try it (free, nothing to install to start): https://nutritional-tracker-delta.vercel.app

Some of what is in it:
- Photo label scanning plus a chat that knows your food log and 7-day averages
- Folders and one-tap meal templates
- Push notifications with meal reminders at times you choose
- Postgres row-level security so users' data is isolated at the database layer

Stack for the curious: React 19, FastAPI, Neon Postgres, Supabase auth, on Vercel and Render. Code is open:
https://github.com/MrTig-afk/NutritionalTracker

It is early and there are not many users yet, so I would genuinely appreciate people trying it and telling me what is broken or confusing. Roast it.
