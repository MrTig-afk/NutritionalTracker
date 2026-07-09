/**
 * demoBoot.js — PUBLIC DEMO build bootstrap (loaded only when VITE_DEMO=1).
 *
 * - Loads the self-hosted font subsets (vite.config strips the Google Fonts
 *   links from index.html in demo builds, so no request leaves the origin).
 * - Tags <body> with .is-demo, which App.jsx uses to show the demo badge.
 */
import "./fonts/fonts.css";

document.body.classList.add("is-demo");
