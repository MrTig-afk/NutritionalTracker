/**
 * pushDemo.js — replaces lib/push.js in the PUBLIC DEMO build (VITE_DEMO=1).
 *
 * The demo registers no service worker, so web push cannot work; reporting
 * "unsupported" makes SettingsTab render its graceful fallback instead of a
 * toggle that would hang on navigator.serviceWorker.ready forever.
 */

export function pushSupported() { return false; }

export function getPermission() { return "default"; }

export async function getSubscribed() { return false; }

export async function enablePush() {
  throw new Error("Push notifications are disabled in the demo build.");
}

export async function disablePush() {}
