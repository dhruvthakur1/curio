import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';
import { Blobs, Footer, Header } from '../components/Layout.jsx';
import Cover from '../components/Cover.jsx';
import { pct } from '../format.js';

// Real output from the trained model for a shopper who rated one game:
// The Legend of Zelda: Breath of the Wild, 5 stars. Splits are each signal's share of the score.
const HERO_INPUT = 'Breath of the Wild';
const HERO_TERMS = ['hyrule', 'sheikah', 'weapons & armor', 'elixirs'];
const HERO_PICKS = [
  { id: 'B087SHFL9B', title: 'Super Mario Odyssey', score: 0.81, als: 0.98, image: 'https://m.media-amazon.com/images/I/61z-eOR5iML.jpg' },
  { id: 'B073K6WMJ9', title: 'Splatoon 2: Starter Pack', score: 0.60, als: 0.99, image: 'https://m.media-amazon.com/images/I/51uIINyrfOL.jpg' },
  { id: 'B09MGJXGBF', title: 'Breath of the Wild (Wii U)', score: 0.55, als: 0.63, image: 'https://m.media-amazon.com/images/I/51L6Hx0HhYL.jpg' },
];

// Shown until the API answers (it can take a moment to wake up on a free host).
const FALLBACK_STATS = {
  dataset: { interactions: 814533, items: 25611, users: 94762 },
  k: 10,
  alpha: 0.8,
  lift_vs_popularity: 2.98,
  coverage_gain_vs_als: 0.387,
  test: {
    hybrid: { 'hit_rate@10': 0.08036, 'coverage@10': 0.18937 },
    als: { 'hit_rate@10': 0.07892, 'coverage@10': 0.13654 },
    tfidf: { 'hit_rate@10': 0.02328, 'coverage@10': 0.76315 },
    popularity: { 'hit_rate@10': 0.027, 'coverage@10': 0.0007 },
  },
};

const AVATARS = [
  ['AK', '#0F766E,#2DD4BF'], ['RM', '#115E59,#14B8A6'], ['SJ', '#0D9488,#5EEAD4'], ['PV', '#134E4A,#2DD4BF'], ['NT', '#0F766E,#99F6E4'],
];

function FusionHero() {
  return (
    <div className="fusion rise d2" role="img"
      aria-label={`Diagram: shoppers who also rated ${HERO_INPUT} (ALS) and keywords from its description (TF-IDF) flow into a fusion step that outputs ranked recommendations: ${HERO_PICKS.map((p) => p.title).join(', ')}`}>
      <svg className="fusion-svg" viewBox="0 0 1120 420" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="outGrad" x1="0" x2="1"><stop offset="0" stopColor="#14B8A6" /><stop offset="1" stopColor="#F59E0B" /></linearGradient>
        </defs>
        <path id="pAls" className="flow-als" d="M302 108 C 420 108, 430 210, 476 210" />
        <path id="pTf" className="flow-tf" d="M302 312 C 420 312, 430 210, 476 210" />
        <path id="pOut1" className="flow-out" d="M644 210 C 700 210, 710 128, 772 128" />
        <path id="pOut2" className="flow-out" d="M644 210 L 772 210" />
        <path id="pOut3" className="flow-out" d="M644 210 C 700 210, 710 292, 772 292" />
        {[['pAls', '#0F766E', 2.2, 0], ['pAls', '#0F766E', 2.2, 1.1], ['pTf', '#D97706', 2.6, 0], ['pTf', '#D97706', 2.6, 1.3],
          ['pOut1', '#0F1A17', 1.8, 0], ['pOut2', '#0F1A17', 1.8, 0.6], ['pOut3', '#0F1A17', 1.8, 1.2]].map(([p, c, d, b], i) => (
          <circle key={i} r="5" fill={c}>
            <animateMotion dur={`${d}s`} begin={`${b}s`} repeatCount="indefinite"><mpath href={`#${p}`} /></animateMotion>
          </circle>
        ))}
      </svg>

      <div className="glass fz-group" style={{ top: 26 }}>
        <div className="fz-head">
          <strong style={{ color: 'var(--teal-ink)' }}>Shoppers like you</strong>
          <span className="chip chip-als" style={{ fontSize: 11 }}>ALS</span>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {AVATARS.map(([t, g], i) => (
            <span key={t} className="avatar" style={{ background: `linear-gradient(135deg,${g})`, animationDelay: `${-0.6 * i}s` }}>{t}</span>
          ))}
        </div>
        <div style={{ fontSize: 12, color: 'var(--ink-2)', marginTop: 10 }}>also rated <strong>{HERO_INPUT}</strong> ★★★★★</div>
      </div>

      <div className="glass fz-group" style={{ bottom: 26 }}>
        <div className="fz-head">
          <strong style={{ color: 'var(--amber-ink)' }}>What it’s about</strong>
          <span className="chip chip-tfidf" style={{ fontSize: 11 }}>TF-IDF</span>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {HERO_TERMS.map((t, i) => <span key={t} className="kw" style={{ animationDelay: `${-i}s` }}>{t}</span>)}
        </div>
      </div>

      <div className="fz-core">
        <div>
          <div style={{ fontFamily: 'var(--display)', fontWeight: 700, fontSize: 18 }}>Fusion</div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-2)', marginTop: 4, lineHeight: 1.45 }}>
            α·<span style={{ color: 'var(--teal)', fontWeight: 700 }}>ALS</span><br />+ (1−α)·<span style={{ color: 'var(--amber-text)', fontWeight: 700 }}>TF-IDF</span>
          </div>
        </div>
      </div>

      <div className="fz-out">
        {HERO_PICKS.map((p, i) => (
          <div key={p.id} className="glass fz-card" style={{ animationDelay: `${i * 0.35}s` }}>
            <Cover item={p} />
            <span style={{ flex: 1, minWidth: 0 }}>
              <span style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 14 }}>
                <strong style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{i + 1} · {p.title}</strong>
                <span className="muted num">{p.score.toFixed(2)}</span>
              </span>
              <span className="split" style={{ height: 7, marginTop: 7 }}>
                <span className="fz-fill" style={{ width: `${p.als * 100}%`, background: 'var(--teal)', animationDelay: `${i * 0.35}s` }} />
                <span className="fz-fill" style={{ width: `${(1 - p.als) * 100}%`, background: 'var(--amber)', animationDelay: `${i * 0.35}s` }} />
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

const ICONS = {
  als: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><circle cx="9" cy="8" r="3.5" /><circle cx="17" cy="9" r="2.5" /><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6M15 14.5c2.8 0 5 2.2 5 5" /></svg>,
  tfidf: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M4 6h16M4 12h10M4 18h7" /><circle cx="18" cy="16" r="3" /><path d="m20.2 18.2 1.8 1.8" /></svg>,
  fuse: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M6 4v6a6 6 0 0 0 12 0V4" /><path d="M12 16v4M8 20h8" /></svg>,
};

export default function Landing() {
  const [stats, setStats] = useState(FALLBACK_STATS);
  useEffect(() => { api.stats().then(setStats).catch(() => {}); }, []);

  const k = stats.k;
  const hit = (m) => stats.test[m][`hit_rate@${k}`];
  const maxHit = Math.max(...['popularity', 'tfidf', 'als', 'hybrid'].map(hit));
  const rows = [
    ['Popularity', 'popularity', 'var(--slate)'],
    ['TF-IDF only', 'tfidf', 'var(--amber)'],
    ['ALS only', 'als', 'var(--teal)'],
    ['Hybrid', 'hybrid', 'linear-gradient(90deg, #0F766E, #F59E0B)'],
  ];

  return (
    <div className="page">
      <Blobs three />
      <Header />
      <main>
        <section className="wrap hero">
          <div className="hero-copy rise">
            <span className="glass pill"><span className="pill-dot" />Hybrid recommender · ALS + TF-IDF</span>
            <h1>Two signals in. <span className="grad">One honest recommendation out.</span></h1>
            <p className="hero-lede">Curio blends what similar shoppers loved with what a game is actually about — and shows how much of each went into every pick.</p>
            <div className="hero-ctas">
              <Link className="btn btn-primary btn-lg" to="/discover">See recommendations</Link>
              <Link className="btn btn-ghost btn-lg" to="/lab">Open the model lab</Link>
            </div>
          </div>
          <FusionHero />
          <p className="muted" style={{ textAlign: 'center', fontSize: 14, margin: '22px 0 0' }}>
            Real output for a shopper who rated one game · Built on public Amazon Video Games reviews
          </p>
        </section>

        <section className="wrap rise d2">
          <div className="glass stats">
            <div><div className="stat-value num">{stats.dataset.interactions.toLocaleString('en')}</div><div className="stat-label">ratings learned from</div></div>
            <div><div className="stat-value num">{stats.dataset.items.toLocaleString('en')}</div><div className="stat-label">games &amp; gear in the catalogue</div></div>
            <div><div className="stat-value num">{stats.dataset.users.toLocaleString('en')}</div><div className="stat-label">shoppers modelled</div></div>
            <div><div className="stat-value num" style={{ color: 'var(--teal)' }}>{stats.lift_vs_popularity.toFixed(1)}×</div><div className="stat-label">hit rate vs. “most popular”</div></div>
          </div>
        </section>

        <section id="how" className="wrap" style={{ paddingTop: 96, paddingBottom: 40, scrollMarginTop: 80 }}>
          <span className="eyebrow">How it works</span>
          <h2 className="section-title">Two models, one ranked list.</h2>
          <div className="how-grid">
            <article className="glass lift how-card">
              <div className="how-icon" style={{ background: 'var(--teal-soft)', color: 'var(--teal)' }}>{ICONS.als}</div>
              <h3>Collaborative · ALS</h3>
              <p>Alternating Least Squares factorises the shopper × product rating matrix to find people with tastes like yours — “fans of this also bought…”.</p>
            </article>
            <article className="glass lift how-card">
              <div className="how-icon" style={{ background: 'var(--amber-soft)', color: 'var(--amber-text)' }}>{ICONS.tfidf}</div>
              <h3>Content · TF-IDF</h3>
              <p>Titles, features and descriptions become TF-IDF vectors, so products are matched by what they’re about — even ones few people have rated.</p>
            </article>
            <article className="glass lift how-card" style={{ background: 'linear-gradient(150deg, rgba(221,242,238,.85), rgba(253,239,214,.8))' }}>
              <div className="how-icon" style={{ background: 'var(--ink)', color: '#fff' }}>{ICONS.fuse}</div>
              <h3>Weighted fusion</h3>
              <p>Both scores are normalised per shopper and blended. The weight, α = {stats.alpha}, was tuned on held-out data. New shoppers start from popularity.</p>
              <code className="formula">score = α·<span style={{ color: 'var(--teal)', fontWeight: 700 }}>ALS</span> + (1−α)·<span style={{ color: 'var(--amber-text)', fontWeight: 700 }}>TF-IDF</span></code>
            </article>
          </div>
        </section>

        <section className="wrap" style={{ paddingTop: 72, paddingBottom: 72, display: 'flex', flexWrap: 'wrap', gap: 56, alignItems: 'center' }}>
          <div style={{ flex: '1 1 420px', minWidth: 0 }}>
            <span className="eyebrow">Measured, not guessed</span>
            <h2 style={{ fontSize: 'clamp(30px, 3.6vw, 42px)', lineHeight: 1.1, margin: '14px 0 18px' }}>Every model is scored on the same held-out test.</h2>
            <p style={{ fontSize: 17, lineHeight: 1.65, color: 'var(--ink-2)', margin: '0 0 26px' }}>
              Each shopper’s most recent purchase was hidden, and every model had to find it among all {stats.dataset.items.toLocaleString('en')} products.
              The hybrid matches ALS on accuracy while recommending <strong>{Math.round(stats.coverage_gain_vs_als * 100)}% more of the catalogue</strong>.
            </p>
            <Link className="btn btn-ghost" to="/lab">Explore the metrics</Link>
          </div>
          <div className="glass" style={{ flex: '1 1 440px', minWidth: 0, padding: 28 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 18 }}>
              <strong style={{ fontFamily: 'var(--display)', fontSize: 18 }}>Hit Rate@{k}</strong>
              <span className="muted" style={{ fontSize: 12 }}>Test set · 25,000 shoppers</span>
            </div>
            <div className="metric-rows">
              {rows.map(([label, key, bg]) => (
                <div key={key}>
                  <div className="metric-row-head">
                    {key === 'hybrid' ? <strong>{label}</strong> : <span>{label}</span>}
                    <span className={`num ${key === 'hybrid' ? '' : 'muted'}`} style={key === 'hybrid' ? { color: 'var(--teal)', fontWeight: 700 } : undefined}>{pct(hit(key), 2)}</span>
                  </div>
                  <div className="metric-track"><div className="bar" style={{ width: `${(hit(key) / maxHit) * 100}%`, background: bg }} /></div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="wrap" style={{ paddingTop: 40, paddingBottom: 104 }}>
          <div className="glass" style={{ borderRadius: 32, padding: 'clamp(32px, 5vw, 64px) clamp(24px, 5vw, 56px)', display: 'flex', flexWrap: 'wrap', gap: 28, alignItems: 'center', justifyContent: 'space-between', background: 'linear-gradient(135deg, rgba(221,242,238,.9), rgba(255,255,255,.7) 50%, rgba(253,239,214,.9))' }}>
            <div style={{ flex: '1 1 420px' }}>
              <h2 style={{ fontSize: 'clamp(30px, 3.6vw, 42px)', lineHeight: 1.1 }}>See why you’d like it <span className="grad">before you buy it.</span></h2>
              <p style={{ color: 'var(--ink-2)', fontSize: 17, margin: '14px 0 0' }}>Pick a sample shopper, or start fresh and rate a few games.</p>
            </div>
            <Link className="btn btn-primary btn-lg" to="/discover">Try Curio</Link>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
