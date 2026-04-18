/**
 * Central API service that injects Authorization headers and projectId
 * into every request so callers don't have to think about auth.
 */

const API_BASE = 'http://localhost:3000/api';

let _getToken = () => null;
let _getProjectId = () => null;

/** Call once at app startup to wire up token/project getters. */
export function configureApiService({ getToken, getProjectId }) {
  _getToken = getToken;
  _getProjectId = getProjectId;
}

/** Build common fetch headers (JSON content type + auth). */
export function authHeaders(extra = {}) {
  const token = _getToken();
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  };
}

/**
 * Append ?projectId=<id> to a URL if a current project is selected
 * and the URL doesn't already include it.
 */
export function withProjectId(url) {
  const projectId = _getProjectId();
  if (!projectId) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}projectId=${projectId}`;
}

/** Authenticated JSON fetch. Throws on non-ok responses. */
export async function apiFetch(path, options = {}) {
  const token = _getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!res.ok) {
    let msg = `API error ${res.status}`;
    try { const d = await res.json(); msg = d.error || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res;
}

/**
 * Upload a video file scoped to the current project.
 * Wraps the existing /api/upload route.
 */
export async function uploadProjectVideo(blob, filename = 'clip.mp4') {
  const formData = new FormData();
  formData.append('video', blob, filename);

  const projectId = _getProjectId();
  const token = _getToken();

  const url = withProjectId(`${API_BASE}/upload`);
  const res = await fetch(url, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });

  if (!res.ok) {
    let msg = `Upload error ${res.status}`;
    try { const d = await res.json(); msg = d.error || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

/**
 * Submit a composition job scoped to the current project.
 * Equivalent to POST /api/process-videos?projectId=...
 */
export async function submitComposeJob(formData) {
  const projectId = _getProjectId();
  const token = _getToken();

  const url = withProjectId(`${API_BASE}/process-videos`);
  const res = await fetch(url, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });

  if (!res.ok) {
    let msg = `Compose error ${res.status}`;
    try { const d = await res.json(); msg = d.error || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

export { API_BASE };
