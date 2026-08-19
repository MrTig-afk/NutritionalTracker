# Security policy

## Reporting a vulnerability

Email theimpracticalguy007@gmail.com. Please include steps to reproduce and
what you think the impact is. You will get a reply within 7 days and a fix or
a status update within 30. Please do not open a public issue for security
problems.

The same contact is published at
`https://nutritional-tracker-delta.vercel.app/.well-known/security.txt`.

## Scope

- The API at `nutritionaltracker.onrender.com`
- The web app at `nutritional-tracker-delta.vercel.app`

Out of scope: Supabase, Neon, Render, Vercel, Google and Groq themselves.

## What is already in place

Per-user row-level security in Postgres, server-side JWT verification on
every route, per-user and global rate limits on paid calls, upload size caps,
per-IP abuse blocking, and a 30-day recycle bin for deleted data.
