import { Link } from 'react-router-dom';
import { Blobs, Footer, Header } from '../components/Layout.jsx';

export default function Placeholder({ title, notFound = false }) {
  return (
    <div className="page">
      <Blobs />
      <Header search />
      <main className="wrap wrap-wide" style={{ padding: '96px 40px' }}>
        <div className="glass" style={{ padding: 40, maxWidth: 620 }}>
          <span className="eyebrow">{notFound ? '404' : 'Coming next'}</span>
          <h1 style={{ fontSize: 36, margin: '12px 0' }}>{title}</h1>
          <p className="muted" style={{ lineHeight: 1.6 }}>
            {notFound ? 'That page doesn’t exist.' : 'This screen is being built in the next phase.'}
          </p>
          <Link className="btn btn-primary btn-sm" to="/discover" style={{ marginTop: 12 }}>Back to recommendations</Link>
        </div>
      </main>
      <Footer />
    </div>
  );
}
