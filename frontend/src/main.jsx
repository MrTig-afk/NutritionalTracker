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
    // Check for update whenever app becomes visible
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') reg.update().catch(() => {});
    });

    // Check for updates every 60 seconds while app is open
    setInterval(() => reg.update().catch(() => {}), 60 * 1000);

    reg.addEventListener('updatefound', () => {
      const newSW = reg.installing;
      newSW.addEventListener('statechange', () => {
        if (newSW.state === 'activated') notifyUpdate();
      });
    });
  }).catch(() => {});
}
