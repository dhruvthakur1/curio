import './styles.css';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import Landing from './pages/Landing.jsx';
import Discover from './pages/Discover.jsx';
import Placeholder from './pages/Placeholder.jsx';
import Product from './pages/Product.jsx';
import Lab from './pages/Lab.jsx';

function ScrollToTop() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [pathname, hash]);
  return null;
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/discover" element={<Discover />} />
        <Route path="/item/:id" element={<Product />} />
        <Route path="/lab" element={<Lab />} />
        <Route path="*" element={<Placeholder title="Page not found" notFound />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
