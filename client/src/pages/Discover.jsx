import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Search, RotateCcw } from 'lucide-react';
import { api } from '../api.js';
import { Blobs, Footer, Header, useSearch } from '../components/Layout.jsx';
import Cover from '../components/Cover.jsx';
import { distinctTerms, initials, itemMeta, shortTitle } from '../format.js';
import { prefs } from '../prefs.js';

const AVATAR = {
  teal: 'linear-gradient(135deg,#0F766E,#2DD4BF)',
  amber: 'linear-gradient(135deg,#C2410C,#F59E0B)',
  slate: 'linear-gradient(135deg,#475569,#94A3B8)',
};

function personaSub(p, ratedCount) {
  if (p.cold) return ratedCount ? `You’ve rated ${ratedCount} game${ratedCount > 1 ? 's' : ''}` : 'No ratings yet — cold start';
  const loved = [];
  for (const h of [...p.history].sort((a, b) => b.rating - a.rating)) {
    const name = shortTitle(h.title.split(':')[0], 22);
    if (!loved.includes(name)) loved.push(name);
    if (loved.length === 3) break;
  }
  return `${p.history.length} real ratings · loves ${loved.join(', ')}`;
}

function Stars({ title, value, onRate }) {
  return (
    <div className="stars" role="group" aria-label={`Rate ${title}`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} type="button" className={`star ${value >= n ? 'on' : ''}`}
          aria-label={`${n} star${n > 1 ? 's' : ''}`} aria-pressed={value === n}
          onClick={() => onRate(value === n ? 0 : n)}>★</button>
      ))}
    </div>
  );
}

function RateItem({ item, value, onRate }) {
  return (
    <div className="cold-item">
      <Cover item={item} />
      <div style={{ minWidth: 0 }}>
        <div className="cold-title" title={item.title}>{shortTitle(item.title, 60)}</div>
        <div className="muted" style={{ fontSize: 12 }}>{item.platform || ' '}</div>
        <Stars title={item.title} value={value} onRate={onRate} />
      </div>
    </div>
  );
}

function ColdStart({ ratings, onRate, onClear }) {
  const [games, setGames] = useState([]);
  const [q, setQ] = useState('');
  const { results, loading, q: searched } = useSearch(q, 8);
  useEffect(() => { api.onboarding(8).then(setGames).catch(() => {}); }, []);

  const shown = q.trim().length >= 2 ? results : [
    ...Object.values(ratings).map((r) => r.item).filter((it) => !games.some((g) => g.id === it.id)),
    ...games,
  ];
  const count = Object.keys(ratings).length;

  return (
    <section className="glass cold rise" aria-labelledby="cold-title">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', alignItems: 'baseline' }}>
        <h2 id="cold-title" style={{ fontSize: 22 }}>Rate a few games to personalise</h2>
        <span className="muted" style={{ fontSize: 14 }}>
          {count ? `${count} rated — every rating re-ranks your picks instantly` : 'Until you rate something you’ll see what’s popular'}
        </span>
      </div>
      <div className="cold-search">
        <label className="search-box">
          <Search size={18} aria-hidden="true" />
          <span className="sr-only">Find a game you’ve played</span>
          <input type="search" placeholder="Find a game you’ve played…" value={q} onChange={(e) => setQ(e.target.value)} />
        </label>
        {count > 0 && (
          <button type="button" className="link-btn" onClick={onClear}>
            <RotateCcw size={14} style={{ verticalAlign: '-2px', marginRight: 6 }} aria-hidden="true" />Clear my ratings
          </button>
        )}
      </div>
      <div className="cold-grid">
        {shown.map((item) => (
          <RateItem key={item.id} item={item} value={ratings[item.id]?.rating || 0}
            onRate={(n) => onRate(item, n)} />
        ))}
        {q.trim().length >= 2 && !results.length && (
          <p className="muted" style={{ margin: 0 }}>{loading || searched !== q.trim() ? 'Searching…' : 'No products match that.'}</p>
        )}
      </div>
    </section>
  );
}

function RecCard({ r, rank, mode, index }) {
  const total = (r.als ?? 0) + (r.tfidf ?? 0);
  const alsW = mode === 'hybrid' && total > 0 ? (r.als / total) * 100 : 0;
  const tfW = mode === 'hybrid' && total > 0 ? (r.tfidf / total) * 100 : 0;
  const popW = mode === 'hybrid' ? 0 : 100;
  const chips = [];
  if (mode !== 'hybrid') chips.push(['chip-pop', 'Popular now']);
  else {
    // Only credit a signal that actually moved the score (≥ 5% of it).
    const als = r.because && alsW >= 5 ? ['chip-als', `Fans of ${shortTitle(r.because.title, 30)}`] : null;
    const tf = r.terms?.length && tfW >= 5 ? ['chip-tfidf', `“${r.terms[0]}”`] : null;
    const ordered = r.als >= r.tfidf ? [als, tf] : [tf, als];
    ordered.filter(Boolean).forEach((c) => chips.push(c));
    // Skip a second term that just repeats the first ("nes" / "nintendo nes").
    const extra = r.terms ? distinctTerms(r.terms, 2)[1] : null;
    if (extra && tf && chips.length < 3) chips.push(['chip-tfidf', `“${extra}”`]);
  }
  const label = mode === 'hybrid'
    ? `Score ${Math.round(alsW)}% collaborative, ${Math.round(tfW)}% content`
    : 'Score from popularity';

  return (
    <article className="glass rec" style={{ animationDelay: `${Math.min(index, 8) * 0.04}s` }}>
      <Link to={`/item/${r.id}`} aria-label={`${r.title} details`} tabIndex={-1}><Cover item={r} label /></Link>
      <div className="rec-body">
        <div className="rec-top"><span>#{rank}</span><span className="rec-score num" title="Hybrid score">{r.score.toFixed(2)}</span></div>
        <h3><Link to={`/item/${r.id}`} title={r.title}>{shortTitle(r.title, 70)}</Link></h3>
        <div className="rec-meta">{itemMeta(r)}</div>
        <div className="split" role="img" aria-label={label} title={label}>
          <span style={{ width: `${alsW}%`, background: 'var(--teal)' }} />
          <span style={{ width: `${tfW}%`, background: 'var(--amber)' }} />
          <span style={{ width: `${popW}%`, background: 'var(--slate)' }} />
        </div>
        <div className="reasons">
          {chips.map(([cls, text]) => <span key={text} className={`chip ${cls}`}><span>{text}</span></span>)}
        </div>
      </div>
    </article>
  );
}

export default function Discover() {
  const [personas, setPersonas] = useState([]);
  const [personaId, setPersonaId] = useState(prefs.persona);
  const [tuned, setTuned] = useState(0.8);
  const [alpha, setAlpha] = useState(prefs.alpha);
  const [gamesOnly, setGamesOnly] = useState(true);
  const [ratings, setRatings] = useState(prefs.ratings);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const ctrlRef = useRef(null);

  useEffect(() => {
    api.personas().then(setPersonas).catch((e) => setError(e.message));
    api.stats().then((s) => { setTuned(s.alpha); setAlpha((a) => a ?? s.alpha); }).catch(() => setAlpha((a) => a ?? 0.8));
  }, [retry]);

  const persona = personas.find((p) => p.id === personaId);
  useEffect(() => {
    if (personas.length && !personas.some((p) => p.id === personaId)) setPersonaId(personas[0].id);
  }, [personas, personaId]);
  const ratingList = useMemo(
    () => Object.values(ratings).map((r) => ({ item_id: r.item.id, rating: r.rating })),
    [ratings],
  );

  useEffect(() => {
    if (!persona || alpha == null) return undefined;
    const body = {
      persona: persona.cold ? undefined : persona.id,
      ratings: persona.cold ? ratingList : [],
      alpha, k: 12, kinds: gamesOnly ? ['game'] : undefined,
    };
    const t = setTimeout(() => {
      ctrlRef.current?.abort();
      const ctrl = new AbortController();
      ctrlRef.current = ctrl;
      setLoading(true);
      api.recommend(body, ctrl.signal)
        .then((res) => { setResult(res); setError(''); setLoading(false); })
        .catch((e) => { if (e.name !== 'AbortError') { setError(e.message); setLoading(false); } });
    }, 140);
    return () => clearTimeout(t);
  }, [persona, alpha, gamesOnly, ratingList]);

  const rate = useCallback((item, n) => {
    setRatings((prev) => {
      const next = { ...prev };
      if (n) next[item.id] = { rating: n, item }; else delete next[item.id];
      prefs.setRatings(next);
      return next;
    });
  }, []);
  const clear = () => { setRatings({}); prefs.setRatings({}); };
  useEffect(() => { prefs.setPersona(personaId); }, [personaId]);
  useEffect(() => { if (alpha != null) prefs.setAlpha(alpha); }, [alpha]);

  const alphaPct = Math.round((alpha ?? tuned) * 100);
  const mode = result?.mode;
  const listTitle = mode === 'popularity' ? 'Popular right now'
    : persona ? `Top picks for ${persona.cold ? 'you' : persona.name.toLowerCase()}` : 'Top picks';
  const listNote = mode === 'popularity' ? 'Cold start: falling back to popularity'
    : `Ranked by the hybrid score · α = ${(alpha ?? tuned).toFixed(2)}`;

  return (
    <div className="page">
      <Blobs />
      <Header search />
      <main className="wrap wrap-wide discover">
        <div className="rise">
          <span className="eyebrow">Recommended for you</span>
          <h1>Pick a shopper, see what Curio picks.</h1>
        </div>

        <fieldset className="personas">
          <legend className="sr-only">Choose a sample shopper</legend>
          {(personas.length ? personas : [{ id: 'a' }, { id: 'b' }, { id: 'c' }]).map((p) => (
            p.name ? (
              <button key={p.id} type="button" className="persona" aria-pressed={p.id === personaId} onClick={() => setPersonaId(p.id)}>
                <span className="persona-avatar" style={{ background: AVATAR[p.color] || AVATAR.slate }}>{initials(p.name)}</span>
                <span style={{ minWidth: 0 }}>
                  <span className="persona-name">{p.name}</span>
                  <span className="persona-sub">{personaSub(p, Object.keys(ratings).length)}</span>
                </span>
              </button>
            ) : <div key={p.id} className="skeleton" style={{ minHeight: 78, borderRadius: 18 }} />
          ))}
        </fieldset>

        {persona && !persona.cold && (
          <details className="glass" style={{ marginTop: 12, padding: '14px 20px' }}>
            <summary style={{ cursor: 'pointer', fontSize: 14, fontWeight: 600, minHeight: 24 }}>
              {persona.blurb} <span className="muted" style={{ fontWeight: 500 }}>· See their {persona.history.length} real ratings</span>
            </summary>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 14 }}>
              {persona.history.map((h) => (
                <Link key={h.id} to={`/item/${h.id}`} className="chip chip-pop" title={h.title} style={{ fontWeight: 500 }}>
                  <span>{shortTitle(h.title, 30)}</span><span style={{ color: 'var(--amber-2)' }}>{'★'.repeat(Math.round(h.rating))}</span>
                </Link>
              ))}
            </div>
          </details>
        )}

        <section className="glass controls">
          <div style={{ flex: '1 1 360px', minWidth: 0 }}>
            <label htmlFor="alpha" className="alpha-label">
              <span>Fusion weight α</span>
              <span><span style={{ color: 'var(--teal)' }}>ALS {alphaPct}%</span> · <span style={{ color: 'var(--amber-text)' }}>TF-IDF {100 - alphaPct}%</span></span>
            </label>
            <input id="alpha" type="range" min="0" max="100" step="5" value={alphaPct}
              onChange={(e) => setAlpha(Number(e.target.value) / 100)}
              aria-valuetext={`ALS ${alphaPct} percent, TF-IDF ${100 - alphaPct} percent`} />
            <div className="range-ticks">
              <span>All content (TF-IDF)</span>
              <button type="button" className="link-btn" style={{ padding: 0, minHeight: 0, fontSize: 12 }}
                onClick={() => setAlpha(tuned)} disabled={alphaPct === Math.round(tuned * 100)}>
                Tuned: {Math.round(tuned * 100)}%
              </button>
              <span>All collaborative (ALS)</span>
            </div>
          </div>
          <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: 6 }}>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.55, color: 'var(--ink-2)' }}>
              Slide to change how much each model counts. The bar on every card shows where its score came from.
            </p>
            <label className="toggle">
              <input type="checkbox" checked={!gamesOnly} onChange={(e) => setGamesOnly(!e.target.checked)} />
              Include consoles &amp; accessories
            </label>
          </div>
        </section>

        {persona?.cold && <ColdStart ratings={ratings} onRate={rate} onClear={clear} />}

        <div className="list-head">
          <h2>{listTitle}</h2>
          <span className="muted" style={{ fontSize: 14 }} aria-live="polite">{loading && result ? 'Updating…' : listNote}</span>
        </div>

        {error && (
          <div className="glass notice" role="alert">
            {error}
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setError(''); setRetry((n) => n + 1); }}>Try again</button>
          </div>
        )}

        <div className={`rec-grid ${loading && result ? 'loading' : ''}`} aria-busy={loading}>
          {result
            ? result.items.map((r, i) => <RecCard key={`${r.id}`} r={r} rank={i + 1} mode={mode} index={i} />)
            : !error && Array.from({ length: 6 }, (_, i) => <div key={i} className="skeleton" />)}
        </div>
      </main>
      <Footer />
    </div>
  );
}
