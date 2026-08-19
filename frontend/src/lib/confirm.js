// In-app replacement for window.confirm/alert. ConfirmHost (components/ConfirmDialog.jsx)
// registers itself here on mount; until then we fall back to the native dialog.
// Usage: `if (!(await confirm("Delete this?"))) return;`
let _open = null;
export const _register = fn => { _open = fn; };
export function confirm(message, { title = "Are you sure?", okLabel = "Delete", cancel = true } = {}) {
  return new Promise(resolve => (_open ? _open({ message, title, okLabel, cancel, resolve }) : resolve(window.confirm(message))));
}
