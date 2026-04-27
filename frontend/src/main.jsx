import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

if ('serviceWorker' in navigator) {
  const notifyUpdate = () => {
    window.__swUpdateReady = true;
    window.dispatchEvent(new CustomEvent('sw-update-ready'));
  };

  navigator.serviceWorker.ready.then(reg => {
    // If a SW is already waiting on load (e.g. user returned to app), show prompt immediately
    if (reg.waiting && navigator.serviceWorker.controller) notifyUpdate();

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') reg.update().catch(() => {});
    });

    setInterval(() => reg.update().catch(() => {}), 60 * 1000);

    reg.addEventListener('updatefound', () => {
      const newSW = reg.installing;
      newSW.addEventListener('statechange', () => {
        // Fire when new SW is installed and waiting — old SW still controls the page
        if (newSW.state === 'installed' && navigator.serviceWorker.controller) notifyUpdate();
      });
    });
  }).catch(() => {});
}
