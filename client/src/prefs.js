// Per-viewer conveniences only: which sample shopper is selected, the α slider,
// and the new shopper's ratings. Every access is guarded so private mode still works.

const KEYS = { persona: 'curio.persona', alpha: 'curio.alpha', ratings: 'curio.newShopperRatings' };

function read(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function write(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

export const prefs = {
  persona: () => read(KEYS.persona, 'retro'),
  setPersona: (id) => write(KEYS.persona, id),
  alpha: () => read(KEYS.alpha, null),
  setAlpha: (a) => write(KEYS.alpha, a),
  ratings: () => read(KEYS.ratings, {}),
  setRatings: (r) => write(KEYS.ratings, r),
};

/** Body for /api/recommend and /api/explain describing the current shopper. */
export function shopperBody(personaId, ratings) {
  if (personaId && personaId !== 'new') return { persona: personaId };
  return { ratings: Object.values(ratings).map((r) => ({ item_id: r.item.id, rating: r.rating })) };
}
