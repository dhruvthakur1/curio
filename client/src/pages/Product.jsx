import { Fragment, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, Users, AlignLeft } from 'lucide-react';
import { api } from '../api.js';
import { Blobs, Footer, Header } from '../components/Layout.jsx';
import Cover from '../components/Cover.jsx';
import { compact, distinctTerms, shortTitle } from '../format.js';
import { prefs, shopperBody } from '../prefs.js';

/** Wrap occurrences of the item's top TF-IDF terms in <mark>. */
function Highlighted({ text, terms }) {
  const parts = useMemo(() => {
    const words = [...new Set(terms.flatMap((t) => [t, ...t.split(' ')]))]
      .filter((w) => w.length > 2)
      .sort((a, b) => b.length - a.length)
      .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/ /g, '\\s+'));
    if (!words.length) return [text];
    const re = new RegExp(`\\b(${words.join('|')})\\b`, 'gi');
    return text.split(re);
  }, [text, terms]);
  return parts.map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : <Fragment key={i}>{p}</Fragment>));
}

function Stars({ value }) {
  const full = Math.round(value);
  return <span aria-hidden="true">{'★'.repeat(full)}<span style={{ opacity: 0.3 }}>{'★'.repeat(5 - full)}</span></span>;
}

function Why({ item, explain, persona, personas, onPersona, alpha }) {
  const name = persona ? (persona.cold ? 'you' : `the ${persona.name.toLowerCase()}`) : 'this shopper';
  const picker = (
    <label className="muted" style={{ fontSize: 13, display: 'inline-flex', gap: 8, alignItems: 'center' }}>
      Explaining for
      <select value={persona?.id || ''} onChange={(e) => onPersona(e.target.value)} className="select">
        {personas.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
      </select>
    </label>
  );

  if (!explain) return <div className="glass why skeleton" style={{ minHeight: 190 }} />;

  if (explain.mode === 'rated') {
    return (
      <div className="glass why">
        <div className="why-head"><h2>Already in {persona?.cold ? 'your' : 'their'} history</h2>{picker}</div>
        <p style={{ margin: '12px 0 0', lineHeight: 1.6 }}>
          {persona?.cold ? 'You' : persona?.name} rated this <strong style={{ color: 'var(--amber-text)' }}>{explain.rating}★</strong>, so Curio never recommends it back —
          it’s one of the inputs that shapes the other picks.
        </p>
      </div>
    );
  }

  if (explain.mode === 'popularity') {
    return (
      <div className="glass why">
        <div className="why-head"><h2>Why Curio would show this</h2>{picker}</div>
        <p style={{ margin: '12px 0 0', lineHeight: 1.6 }}>
          There’s no rating history yet, so Curio falls back to popularity: this is the
          <strong> #{explain.popularity_rank.toLocaleString('en')}</strong> most-liked product in the catalogue.{' '}
          <Link to="/discover">Rate a game</Link> to get a personal explanation.
        </p>
      </div>
    );
  }

  const total = explain.als.weighted + explain.tfidf.weighted || 1;
  const alsPct = Math.round((explain.als.weighted / total) * 100);
  const tfPct = 100 - alsPct;
  const terms = distinctTerms(explain.shared_terms.map((t) => t.term)).map((t) => `“${t}”`);
  return (
    <div className="glass why">
      <div className="why-head">
        <h2>Why Curio picked this for {name}</h2>
        <span className="why-score num">{explain.score.toFixed(2)} <small>hybrid score · #{explain.rank.toLocaleString('en')} of {explain.of.toLocaleString('en')}</small></span>
      </div>
      <div className="split" style={{ height: 12, margin: '16px 0 12px' }} role="img" aria-label={`${alsPct} percent collaborative, ${tfPct} percent content`}>
        <span className="bar" style={{ width: `${alsPct}%`, background: 'var(--teal)' }} />
        <span className="bar" style={{ width: `${tfPct}%`, background: 'var(--amber)', animationDelay: '.5s' }} />
      </div>
      <div className="why-grid">
        <p><strong style={{ color: 'var(--teal-ink)' }}>{alsPct}% collaborative.</strong>{' '}
          {explain.because
            ? <>Shoppers who liked <Link to={`/item/${explain.because.id}`}><em>{shortTitle(explain.because.title, 40)}</em></Link> — like {name} — also bought this.</>
            : <>Shoppers with a similar history bought this.</>}
        </p>
        <p><strong style={{ color: 'var(--amber-ink)' }}>{tfPct}% content.</strong>{' '}
          {terms.length
            ? <>Its description shares distinctive terms with products {persona?.cold ? 'you' : 'they'} liked, like {terms.join(', ')}.</>
            : <>Its description has little in common with {persona?.cold ? 'your' : 'their'} history.</>}
        </p>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginTop: 14 }}>
        {picker}
        <span className="muted" style={{ fontSize: 13 }}>α = {alpha.toFixed(2)} · change it on the <Link to="/discover">For you</Link> page</span>
      </div>
    </div>
  );
}

function Rail({ title, sub, icon, tone, items }) {
  return (
    <div className="glass" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 18 }}>
        <span className="rail-icon" style={{ background: tone === 'als' ? 'var(--teal-soft)' : 'var(--amber-soft)', color: tone === 'als' ? 'var(--teal)' : 'var(--amber-text)' }}>{icon}</span>
        <div><h2 style={{ fontSize: 19 }}>{title}</h2><div className="muted" style={{ fontSize: 13 }}>{sub}</div></div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {items ? items.map((it) => (
          <Link key={it.id} to={`/item/${it.id}`} className="mini">
            <Cover item={it} />
            <span style={{ flex: 1, minWidth: 0 }}>
              <strong className="mini-title" title={it.title}>{shortTitle(it.title, 60)}</strong>
              <span className="muted" style={{ fontSize: 13 }}>{it.platform || it.store || '\u00a0'}</span>
            </span>
            <span className={`chip chip-${tone}`} title="Similarity">{it.similarity.toFixed(2)}</span>
          </Link>
        )) : Array.from({ length: 5 }, (_, i) => <div key={i} className="skeleton" style={{ minHeight: 76, borderRadius: 16 }} />)}
      </div>
    </div>
  );
}

export default function Product() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [item, setItem] = useState(null);
  const [similar, setSimilar] = useState(null);
  const [explain, setExplain] = useState(null);
  const [personas, setPersonas] = useState([]);
  const [personaId, setPersonaId] = useState(prefs.persona);
  const [alpha, setAlpha] = useState(prefs.alpha() ?? 0.8);
  const [error, setError] = useState('');

  useEffect(() => {
    setItem(null); setSimilar(null); setError('');
    api.item(id).then((d) => { setItem(d); document.title = `${shortTitle(d.title)} · Curio`; }).catch((e) => setError(e.message));
    api.similar(id, 5).then(setSimilar).catch(() => {});
  }, [id]);

  useEffect(() => {
    api.personas().then(setPersonas).catch(() => {});
    if (prefs.alpha() == null) api.stats().then((s) => setAlpha(s.alpha)).catch(() => {});
    return () => { document.title = 'Curio · Hybrid game recommender'; };
  }, []);

  useEffect(() => {
    setExplain(null);
    api.explain({ item_id: id, alpha, ...shopperBody(personaId, prefs.ratings()) })
      .then(setExplain).catch(() => setExplain({ mode: 'popularity', popularity_rank: 0 }));
  }, [id, personaId, alpha]);

  const persona = personas.find((p) => p.id === personaId);
  const pickPersona = (pid) => { setPersonaId(pid); prefs.setPersona(pid); };
  const terms = item?.top_terms?.map((t) => t.term) || [];
  const maxW = item?.top_terms?.[0]?.weight || 1;

  return (
    <div className="page">
      <Blobs />
      <Header search />
      <main className="wrap wrap-wide product">
        <button type="button" className="back" onClick={() => (window.history.length > 1 ? navigate(-1) : navigate('/discover'))}>
          <ArrowLeft size={16} aria-hidden="true" /> Back
        </button>

        {error ? (
          <div className="glass notice" role="alert" style={{ marginTop: 16 }}>
            {error === 'Product not found' ? 'That product isn’t in the catalogue.' : error}
            <Link className="btn btn-ghost btn-sm" to="/discover">Back to recommendations</Link>
          </div>
        ) : (
          <>
            <section className="product-top">
              <div className="rise product-cover-wrap">
                {item ? <Cover item={item} className="product-cover" label /> : <div className="skeleton product-cover" />}
              </div>
              <div className="rise d1" style={{ flex: '1 1 520px', minWidth: 0 }}>
                {item ? (
                  <>
                    <span className="eyebrow">{[item.platform, item.store].filter(Boolean).join(' · ') || 'Video Games'}</span>
                    <h1 className="product-title">{shortTitle(item.title, 90)}</h1>
                    {shortTitle(item.title, 90) !== item.title && <p className="muted" style={{ margin: '0 0 8px', fontSize: 14 }}>{item.title}</p>}
                    <div className="product-meta">
                      {item.average_rating && <span style={{ color: 'var(--amber-text)', fontWeight: 700 }}><Stars value={item.average_rating} /> {item.average_rating.toFixed(1)}</span>}
                      {item.rating_count > 0 && <span>{compact(item.rating_count)} Amazon ratings</span>}
                      <span>{item.dataset_ratings.toLocaleString('en')} in Curio’s training data</span>
                      {item.price != null && <span>${item.price.toFixed(2)}</span>}
                      <span className={`chip ${item.kind === 'game' ? 'chip-als' : 'chip-pop'}`} style={{ textTransform: 'capitalize' }}>{item.kind}</span>
                    </div>
                  </>
                ) : <div className="skeleton" style={{ height: 120 }} />}

                <Why item={item} explain={explain} persona={persona} personas={personas} onPersona={pickPersona} alpha={alpha} />

                {item && (
                  <>
                    <h2 style={{ fontSize: 19, margin: '30px 0 10px' }}>About this product</h2>
                    <p className="product-desc">
                      {item.description
                        ? <Highlighted text={item.description} terms={terms} />
                        : <span className="muted">Amazon didn’t include a description, so TF-IDF works from the title, categories and feature list.</span>}
                    </p>
                    {item.top_terms.length > 0 && (
                      <div style={{ marginTop: 22 }}>
                        <div className="muted" style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>
                          Most distinctive TF-IDF terms {item.description && <span style={{ fontWeight: 500 }}>(highlighted above)</span>}
                        </div>
                        <div className="terms-grid">
                          {item.top_terms.map((t) => (
                            <div key={t.term}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}><span>{t.term}</span><span className="muted num">{t.weight.toFixed(2)}</span></div>
                              <div className="term-track"><div className="bar" style={{ width: `${(t.weight / maxW) * 100}%` }} /></div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </section>

            <section className="rise d2 rails">
              <Rail title="Similar by description" sub="TF-IDF cosine similarity" tone="tfidf" icon={<AlignLeft size={20} aria-hidden="true" />} items={similar?.similar_content} />
              <Rail title="Fans also bought" sub="ALS item-factor similarity" tone="als" icon={<Users size={20} aria-hidden="true" />} items={similar?.also_bought} />
            </section>
          </>
        )}
      </main>
      <Footer />
    </div>
  );
}
