import { apiFetch } from "./api";

const VAPID_PUBLIC_KEY = import.meta.env.VITE_VAPID_PUBLIC_KEY;

function urlB64ToUint8Array(b64) {
  const pad = "=".repeat((4 - b64.length % 4) % 4);
  const raw = atob((b64 + pad).replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from([...raw].map(c => c.charCodeAt(0)));
}

export function pushSupported() {
  return "serviceWorker" in navigator && "PushManager" in window && "Notification" in window;
}

export function getPermission() {
  return "Notification" in window ? Notification.permission : "default";
}

export async function getSubscribed() {
  if (!pushSupported()) return false;
  const reg = await navigator.serviceWorker.ready;
  const sub = await reg.pushManager.getSubscription();
  return !!sub;
}

/** Request permission, subscribe this device, register with the backend.
 *  Returns the resulting Notification.permission string. */
export async function enablePush() {
  if (!pushSupported()) throw new Error("Push notifications not supported in this browser.");
  const perm = await Notification.requestPermission();
  if (perm !== "granted") return perm;
  const reg = await navigator.serviceWorker.ready;
  const keyRes = await apiFetch("/push/vapid-key").catch(() => null);
  const publicKey = keyRes?.public_key || VAPID_PUBLIC_KEY;
  if (!publicKey) throw new Error("Push not configured on server yet.");
  const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: urlB64ToUint8Array(publicKey) });
  await apiFetch("/push/subscribe", { method: "POST", body: JSON.stringify(sub) });
  return perm;
}

/** Unsubscribe this device and deregister with the backend. */
export async function disablePush() {
  const reg = await navigator.serviceWorker.ready;
  const sub = await reg.pushManager.getSubscription();
  if (sub) await sub.unsubscribe();
  await apiFetch("/push/unsubscribe", { method: "DELETE" });
}
