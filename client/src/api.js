const BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}/api${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
  } catch {
    throw new Error('Can’t reach the Curio API. It may be waking up — try again in a moment.');
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${res.status})`);
  }
  return body;
}

export const api = {
  stats: () => request('/stats'),
  metrics: () => request('/metrics'),
  personas: () => request('/personas'),
  recommend: (body, signal) => request('/recommend', { method: 'POST', body: JSON.stringify(body), signal }),
  explain: (body) => request('/explain', { method: 'POST', body: JSON.stringify(body) }),
  item: (id) => request(`/items/${encodeURIComponent(id)}`),
  similar: (id, k = 8) => request(`/items/${encodeURIComponent(id)}/similar?k=${k}`),
  search: (q, limit = 8, signal) => request(`/search?q=${encodeURIComponent(q)}&limit=${limit}`, { signal }),
  onboarding: (k = 8) => request(`/onboarding?k=${k}`),
};
