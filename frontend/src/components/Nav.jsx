import { useState } from 'react';
import { Logo } from './common.jsx';

const LINKS = [
  ['home', 'Home'],
  ['upload', 'Upload'],
  ['results', 'Results'],
  ['history', 'History'],
  ['research', 'Research'],
  ['about', 'About'],
];

export default function Nav({ route, navigate, online }) {
  const [open, setOpen] = useState(false);
  const go = (r) => { setOpen(false); navigate(r); };
  const active = route === 'analysis' ? 'upload' : route === 'evidence' ? 'results' : route;
  return (
    <header className="tg-nav">
      <a className="tg-brand" href="#/home" onClick={(e) => { e.preventDefault(); go('home'); }} aria-label="TumorGuard AI home">
        <Logo />
        <span>TumorGuard AI</span>
      </a>
      <button
        type="button"
        className="tg-nav-toggle"
        aria-expanded={open}
        aria-controls="tg-nav-links"
        aria-label={open ? 'Close menu' : 'Open menu'}
        onClick={() => setOpen((o) => !o)}
      >
        <span aria-hidden="true">{open ? '✕' : '☰'}</span>
      </button>
      <nav id="tg-nav-links" className={`tg-nav-links${open ? ' is-open' : ''}`} aria-label="Primary">
        {LINKS.map(([r, label]) => (
          <a
            key={r}
            href={`#/${r}`}
            aria-current={active === r ? 'page' : undefined}
            onClick={(e) => { e.preventDefault(); go(r); }}
          >
            {label}
          </a>
        ))}
        <button type="button" className="tg-btn tg-btn-primary tg-nav-cta" onClick={() => go('upload')}>New analysis</button>
      </nav>
      <span className={`tg-status ${online === true ? 'is-on' : online === false ? 'is-off' : ''}`} role="status">
        <span className="tg-dot" aria-hidden="true" />
        {online === null ? 'Checking server…' : online ? 'Server online' : 'Server offline'}
      </span>
    </header>
  );
}
