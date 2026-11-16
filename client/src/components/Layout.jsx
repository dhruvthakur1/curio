import { useEffect, useRef, useState } from 'react';
import { Link, NavLink, useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { api } from '../api.js';
import { itemMeta, shortTitle } from '../format.js';
import Cover from './Cover.jsx';

export function Logo({ size = 34 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 34 34" aria-hidden="true">
      <rect width="34" height="34" rx="10" fill="#0F1A17" />
      <circle cx="13" cy="14" r="5" fill="#2DD4BF" />
      <circle cx="21" cy="20" r="5" fill="#F59E0B" fillOpacity=".9" />
    </svg>
  );
}

export function Blobs({ three = false }) {
  return (
    <div className="blobs" aria-hidden="true">
      <div className="blob b1" />
      <div className="blob b2" />
      {three && <div className="blob b3" />}
    </div>
  );
}

export function useSearch(q, limit = 8) {
  const [state, setState] = useState({ q: '', results: [], loading: false });
  useEffect(() => {
    const term = q.trim();
    if (term.length < 2) {
      setState({ q: term, results: [], loading: false });
      return undefined;
    }
    const ctrl = new AbortController();
    setState((s) => ({ ...s, loading: true }));
    const t = setTimeout(() => {
      api.search(term, limit, ctrl.signal)
        .then((results) => setState({ q: term, results, loading: false }))
        .catch((e) => { if (e.name !== 'AbortError') setState({ q: term, results: [], loading: false }); });
    }, 220);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [q, limit]);
  return state;
}

function HeaderSearch() {
  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const { results, loading, q: searched } = useSearch(q);
  const navigate = useNavigate();
  const boxRef = useRef(null);

  useEffect(() => {
    const close = (e) => { if (!boxRef.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('pointerdown', close);
    return () => document.removeEventListener('pointerdown', close);
  }, []);
  useEffect(() => setActive(-1), [results]);

  const go = (item) => { setOpen(false); setQ(''); navigate(`/item/${item.id}`); };
  const onKey = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, results.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
    else if (e.key === 'Enter' && results[Math.max(active, 0)]) { e.preventDefault(); go(results[Math.max(active, 0)]); }
    else if (e.key === 'Escape') setOpen(false);
  };
  const show = open && q.trim().length >= 2;

  return (
    <div className="search" ref={boxRef}>
      <label className="search-box">
        <Search size={18} aria-hidden="true" />
        <span className="sr-only">Search products</span>
        <input
          type="search" placeholder="Search 25,000+ games and gear…" value={q}
          onChange={(e) => { setQ(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)} onKeyDown={onKey}
          role="combobox" aria-expanded={show} aria-controls="search-results" aria-autocomplete="list"
        />
      </label>
      {show && (
        <ul id="search-results" className="search-results glass" role="listbox">
          {results.map((item, i) => (
            <li key={item.id} role="option" aria-selected={i === active}>
              <a href={`/item/${item.id}`} className={i === active ? 'on' : ''}
                onClick={(e) => { e.preventDefault(); go(item); }}>
                <Cover item={item} style={{ width: 36, height: 44, borderRadius: 8 }} />
                <span style={{ minWidth: 0 }}>
                  <span style={{ display: 'block', fontWeight: 600 }}>{shortTitle(item.title, 56)}</span>
                  <span className="muted" style={{ fontSize: 12 }}>{itemMeta(item)}</span>
                </span>
              </a>
            </li>
          ))}
          {!results.length && (
            <li className="empty">{loading || searched !== q.trim() ? 'Searching…' : 'No products match that'}</li>
          )}
        </ul>
      )}
    </div>
  );
}

export function Header({ search = false }) {
  return (
    <header className="site-header">
      <div className={`wrap ${search ? 'wrap-wide' : ''}`}>
        <Link to="/" className="brand"><Logo /> Curio</Link>
        {search && <HeaderSearch />}
        <nav className="nav" aria-label="Main">
          {!search && <a className="nav-link hide-sm" href="/#how">How it works</a>}
          <NavLink className="nav-link" to="/discover">For you</NavLink>
          <NavLink className="nav-link" to="/lab">Model lab</NavLink>
          {!search && <a className="nav-link hide-sm" href="https://github.com/dhruvthakur1/curio" target="_blank" rel="noreferrer">GitHub</a>}
          {!search && <Link className="btn btn-primary btn-sm" to="/discover">Try it</Link>}
        </nav>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="wrap">
        <span><strong style={{ color: '#0F1A17' }}>Curio</strong> · A portfolio project by Dhruv Thakur, based on his IBM internship work</span>
        <span>Data: Amazon Reviews 2023 (McAuley Lab, UCSD) · Python · FastAPI · React</span>
      </div>
    </footer>
  );
}
