/** Amazon titles end in " - Platform" or "(Edition)"; keep the game's own name. */
export function shortTitle(title, max = 48) {
  const base = title.split(/\s+[-–]\s+|\s+\(|\s+\[/)[0].trim() || title;
  return base.length <= max ? base : `${base.slice(0, max - 1).trimEnd()}…`;
}

export function compact(n) {
  if (n == null) return '—';
  return new Intl.NumberFormat('en', { notation: n >= 100000 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(n);
}

export function itemMeta(item) {
  const parts = [];
  if (item.platform) parts.push(item.platform);
  if (item.average_rating) parts.push(`★ ${item.average_rating.toFixed(1)}`);
  if (item.rating_count) parts.push(`${compact(item.rating_count)} ratings`);
  return parts.join(' · ');
}

export function initials(name) {
  return name.split(/\s+/).map((w) => w[0]).join('').slice(0, 2).toUpperCase();
}

const COVERS = [
  'linear-gradient(160deg,#134E4A,#0F766E 55%,#5EEAD4)',
  'linear-gradient(160deg,#7C2D12,#C2410C 55%,#FDBA74)',
  'linear-gradient(160deg,#1E3A8A,#2563EB 55%,#93C5FD)',
  'linear-gradient(160deg,#581C87,#9333EA 55%,#D8B4FE)',
  'linear-gradient(160deg,#14532D,#16A34A 55%,#86EFAC)',
  'linear-gradient(160deg,#831843,#DB2777 55%,#F9A8D4)',
];

export function coverGradient(id = '') {
  let h = 0;
  for (const c of id) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return COVERS[h % COVERS.length];
}

export const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
