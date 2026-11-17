import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';
import { Blobs, Footer, Header } from '../components/Layout.jsx';
import { pct } from '../format.js';

const METRICS = {
  hit_rate: {
    label: 'Hit rate', fmt: (v) => pct(v, 2),
    help: 'How often the product the shopper actually bought next was somewhere in the top 10. With one held-out purchase per shopper this is also Recall@10.',
  },
  ndcg: {
    label: 'NDCG', fmt: (v) => v.toFixed(4),
    help: 'Like hit rate, but a hit at #1 counts more than a hit at #10. This is the metric α was tuned on.',
  },
  precision: {
    label: 'Precision', fmt: (v) => pct(v, 2),
    help: 'Of the 10 products recommended, the share that was the real next purchase. With one held-out item it is hit rate ÷ 10.',
  },
  coverage: {
    label: 'Coverage', fmt: (v) => pct(v, 1),
    help: 'How much of the catalogue ever appears in anyone’s top 10. Higher means less popularity bias and more variety.',
  },
};

const MODELS = [
  { key: 'popularity', name: 'Popularity', note: 'baseline', fill: 'var(--slate)', text: 'var(--ink-2)' },
  { key: 'tfidf', name: 'TF-IDF', note: 'content', fill: 'var(--amber)', text: 'var(--amber-ink)' },
  { key: 'als', name: 'ALS', note: 'collaborative', fill: 'var(--teal)', text: 'var(--teal-ink)' },
  { key: 'hybrid', name: 'Hybrid', note: '', fill: 'linear-gradient(180deg, #0F766E, #F59E0B)', text: 'var(--ink)' },
];

function Tabs({ value, onChange, options, label }) {
  return (
    <div role="tablist" aria-label={label} className="tabs">
      {options.map(([key, text]) => (
        <button key={key} type="button" role="tab" aria-selected={value === key} className="tab" onClick={() => onChange(key)}>{text}</button>
      ))}
    </div>
  );
}

function Comparison({ m }) {
  const [metric, setMetric] = useState('hit_rate');
  const def = METRICS[metric];
  const values = MODELS.map((model) => m.test[model.key][`${metric}@${m.k}`]);
  const max = Math.max(...values);
  const best = MODELS[values.indexOf(max)].key;
  return (
    <section className="glass rise d2 lab-card">
      <div className="lab-card-head">
        <h2>Model comparison @{m.k}</h2>
        <Tabs label="Metric" value={metric} onChange={setMetric} options={Object.entries(METRICS).map(([k, d]) => [k, d.label])} />
      </div>
      <p className="lab-help">{def.help}</p>
      <div className="cols" role="img" aria-label={MODELS.map((mo, i) => `${mo.name} ${def.fmt(values[i])}`).join(', ')}>
        {MODELS.map((model, i) => (
          <div key={model.key} className="col-wrap">
            <span className="col-value num" style={{ color: model.text }}>{def.fmt(values[i])}</span>
            <div className="col" key={metric} style={{ height: `${Math.max(2, (values[i] / max) * 100)}%`, background: model.fill, outline: model.key === best ? '2px solid rgba(15,26,23,.12)' : 'none' }} />
          </div>
        ))}
      </div>
      <div className="col-labels">
        {MODELS.map((model) => (
          <div key={model.key}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{model.name}</div>
            <div className="muted" style={{ fontSize: 12 }}>{model.key === 'hybrid' ? `α = ${m.alpha.toFixed(2)}` : model.note}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function AlphaCurve({ m }) {
  const curve = m.alpha_curve;
  const [metric, setMetric] = useState('ndcg');
  const tunedIdx = curve.findIndex((c) => Math.abs(c.alpha - m.alpha) < 1e-6);
  const [idx, setIdx] = useState(tunedIdx >= 0 ? tunedIdx : curve.length - 1);
  const key = `${metric}@${m.k}`;
  const vals = curve.map((c) => c[key]);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = (hi - lo) * 0.12 || 0.001;
  const W = 520, H = 250, L = 56, R = 500, T = 20, B = 200;
  const x = (a) => L + a * (R - L);
  const y = (v) => B - ((v - (lo - pad)) / (hi + pad - (lo - pad))) * (B - T);
  const pts = curve.map((c) => `${x(c.alpha).toFixed(1)},${y(c[key]).toFixed(1)}`).join(' ');
  const ticks = [lo, (lo + hi) / 2, hi];
  const point = curve[idx];
  const fmt = METRICS[metric].fmt;
  const grad = metric === 'coverage' ? ['#0F766E', '#F59E0B'] : ['#F59E0B', '#0F766E'];

  return (
    <section className="glass lab-card">
      <div className="lab-card-head">
        <h2>Tuning the fusion weight</h2>
        <Tabs label="Curve metric" value={metric} onChange={setMetric}
          options={[['ndcg', 'NDCG'], ['hit_rate', 'Hit rate'], ['coverage', 'Coverage']]} />
      </div>
      <p className="lab-help">
        {METRICS[metric].label}@{m.k} on {m.protocol.validation_users_alpha.toLocaleString('en')} validation shoppers as α moves from all-content (0) to all-collaborative (1).
        {metric === 'coverage' && ' Variety falls as ALS takes over — the trade-off the hybrid balances.'}
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="curve-svg" role="img"
        aria-label={`${METRICS[metric].label} by alpha: ${curve.filter((_, i) => i % 4 === 0).map((c) => `alpha ${c.alpha} ${fmt(c[key])}`).join(', ')}`}>
        <defs><linearGradient id="curveGrad" x1="0" x2="1"><stop offset="0" stopColor={grad[0]} /><stop offset="1" stopColor={grad[1]} /></linearGradient></defs>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={R} y1={y(t)} y2={y(t)} stroke="#E3E8E6" />
            <text x={L - 8} y={y(t) + 4} fontSize="11" fill="#55605C" textAnchor="end" fontFamily="Inter, sans-serif">{fmt(t)}</text>
          </g>
        ))}
        <line x1={L} y1={B} x2={R} y2={B} stroke="#D5DDDA" />
        {[0, 0.25, 0.5, 0.75, 1].map((a) => (
          <text key={a} x={x(a)} y={B + 20} fontSize="12" fill="#55605C" textAnchor="middle" fontFamily="Inter, sans-serif">{a === 1 ? '1.0 (α)' : a}</text>
        ))}
        {tunedIdx >= 0 && (
          <g>
            <line x1={x(m.alpha)} x2={x(m.alpha)} y1={T} y2={B} stroke="#0F766E" strokeOpacity=".35" strokeDasharray="2 4" />
            <text x={x(m.alpha)} y={T - 6} fontSize="11" fill="#0F5F58" fontWeight="700" textAnchor="middle" fontFamily="Inter, sans-serif">tuned</text>
          </g>
        )}
        <polyline key={metric} className="curve" pathLength="1" points={pts} fill="none" stroke="url(#curveGrad)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
        <line x1={x(point.alpha)} y1={T} x2={x(point.alpha)} y2={B} stroke="#0F1A17" strokeDasharray="4 5" strokeOpacity=".35" />
        <circle cx={x(point.alpha)} cy={y(point[key])} r="8" fill="#fff" stroke="#0F1A17" strokeWidth="3" style={{ transition: 'cx .25s, cy .25s' }} />
      </svg>
      <label htmlFor="lab-alpha" className="alpha-readout">
        <span>α = {point.alpha.toFixed(2)}</span>
        <span className="num">NDCG {point[`ndcg@${m.k}`].toFixed(4)} · Hit {pct(point[`hit_rate@${m.k}`], 2)} · Coverage {pct(point[`coverage@${m.k}`], 1)}</span>
      </label>
      <input id="lab-alpha" type="range" min="0" max={curve.length - 1} step="1" value={idx}
        onChange={(e) => setIdx(Number(e.target.value))} aria-valuetext={`alpha ${point.alpha.toFixed(2)}`} />
    </section>
  );
}

function Segments({ m }) {
  const k = m.k;
  const rows = m.segments;
  const max = Math.max(...rows.flatMap((s) => MODELS.map((mo) => s[mo.key][`hit_rate@${k}`])));
  const short = rows[0];
  const gain = short.hybrid[`hit_rate@${k}`] / short.als[`hit_rate@${k}`] - 1;
  return (
    <section className="glass lab-card">
      <h2>Short vs. long histories</h2>
      <p className="lab-help">Hit Rate@{k} on the test set, split by how many products each shopper had rated before. Short histories are where ALS has the least to learn from.</p>
      <div className="seg-legend">
        {MODELS.map((mo) => <span key={mo.key}><i style={{ background: mo.fill }} />{mo.name}</span>)}
      </div>
      <div className="segs">
        {rows.map((s) => (
          <div key={s.segment} className="seg">
            <div className="seg-head"><strong>{s.segment}</strong><span className="muted num">{s.users.toLocaleString('en')} shoppers</span></div>
            {MODELS.map((mo) => {
              const v = s[mo.key][`hit_rate@${k}`];
              return (
                <div key={mo.key} className="seg-row" title={`${mo.name}: ${pct(v, 2)}`}>
                  <div className="seg-track"><div className="bar" style={{ width: `${(v / max) * 100}%`, background: mo.fill }} /></div>
                  <span className="num" style={{ fontWeight: mo.key === 'hybrid' ? 700 : 500, color: mo.key === 'hybrid' ? 'var(--teal)' : 'var(--ink-2)' }}>{pct(v, 1)}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <p className="callout">
        <strong>Honest read:</strong> the hybrid edges out ALS in every group ({gain >= 0 ? '+' : ''}{(gain * 100).toFixed(1)}% relative for 3–4 ratings).
        Its bigger win is variety: it recommends {Math.round((m.test.hybrid[`coverage@${k}`] / m.test.als[`coverage@${k}`] - 1) * 100)}% more of the catalogue at the same accuracy.
        Brand-new shoppers with no history start from popularity, and a single rating personalises their picks.
      </p>
    </section>
  );
}

function Grid({ m }) {
  const k = m.k;
  const best = m.als_grid.reduce((a, b) => (b[`ndcg@${k}`] > a[`ndcg@${k}`] ? b : a));
  return (
    <section className="glass lab-card">
      <h2>ALS hyperparameter search</h2>
      <p className="lab-help">Eight configurations scored on {m.protocol.validation_users_grid.toLocaleString('en')} validation shoppers (ALS alone, before fusion). The best one was kept.</p>
      <div className="table-wrap">
        <table className="grid-table">
          <thead><tr><th>Factors</th><th>Regularisation</th><th>Confidence α</th><th>NDCG@{k}</th><th>Hit rate@{k}</th><th>Coverage@{k}</th></tr></thead>
          <tbody>
            {m.als_grid.map((g) => (
              <tr key={`${g.factors}-${g.regularization}-${g.alpha}`} className={g === best ? 'best' : ''}>
                <td>{g.factors}</td><td>{g.regularization}</td><td>{g.alpha}</td>
                <td className="num">{g[`ndcg@${k}`].toFixed(4)}</td><td className="num">{pct(g[`hit_rate@${k}`], 2)}</td><td className="num">{pct(g[`coverage@${k}`], 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function Lab() {
  const [m, setM] = useState(null);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  useEffect(() => { setError(''); api.metrics().then(setM).catch((e) => setError(e.message)); }, [retry]);

  const trained = useMemo(() => (m ? new Date(m.trained_at).toLocaleDateString('en', { day: 'numeric', month: 'short', year: 'numeric' }) : ''), [m]);

  return (
    <div className="page">
      <Blobs />
      <Header search />
      <main className="wrap wrap-wide lab">
        <div className="rise" style={{ display: 'flex', justifyContent: 'space-between', gap: 20, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div>
            <span className="eyebrow">Model lab</span>
            <h1>Does the hybrid actually beat each model on its own?</h1>
          </div>
          {m && <span className="badge"><span className="pill-dot" />Real evaluation · trained {trained}</span>}
        </div>

        {error && (
          <div className="glass notice" role="alert" style={{ marginTop: 24 }}>
            {error}
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setRetry((n) => n + 1)}>Try again</button>
          </div>
        )}

        {!m && !error && <div className="skeleton" style={{ marginTop: 30, minHeight: 420 }} />}

        {m && (
          <>
            <section className="rise d1 facts">
              <div className="glass fact">
                <div className="fact-label">Dataset</div>
                <div className="fact-value">Amazon Video Games</div>
                <div className="fact-sub">{m.dataset.interactions.toLocaleString('en')} ratings · {m.dataset.items.toLocaleString('en')} products · {m.dataset.users.toLocaleString('en')} shoppers</div>
              </div>
              <div className="glass fact">
                <div className="fact-label">Test split</div>
                <div className="fact-value">Each shopper’s latest purchase</div>
                <div className="fact-sub">Train on the past, test on the future · {m.protocol.test_users.toLocaleString('en')} shoppers</div>
              </div>
              <div className="glass fact">
                <div className="fact-label">The task</div>
                <div className="fact-value">Find it in the top {m.k}</div>
                <div className="fact-sub">out of every product they hadn’t rated yet</div>
              </div>
              <div className="glass fact fact-accent">
                <div className="fact-label">Best fusion weight</div>
                <div className="fact-value" style={{ color: 'var(--teal-ink)' }}>α = {m.alpha.toFixed(2)}</div>
                <div className="fact-sub">tuned on separate validation shoppers</div>
              </div>
            </section>

            <Comparison m={m} />

            <div className="rise d3 lab-pair">
              <AlphaCurve m={m} />
              <Segments m={m} />
            </div>

            <div className="rise d3" style={{ marginTop: 20 }}>
              <Grid m={m} />
            </div>

            <p className="muted" style={{ fontSize: 13, lineHeight: 1.6, marginTop: 24, maxWidth: 820 }}>
              Protocol: ratings sorted by time per shopper; the last one is the test item, the one before it is the validation item, the rest is training data.
              ALS settings and α were chosen on validation shoppers only, then ALS was refit on training + validation data and every model was scored once on the test set.
              Absolute hit rates look small because each shopper has exactly one correct answer among {m.dataset.items.toLocaleString('en')} products. <Link to="/discover">Try the model yourself →</Link>
            </p>
          </>
        )}
      </main>
      <Footer />
    </div>
  );
}
