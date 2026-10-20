import { useId, useRef, useEffect } from 'react';

export function Logo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="tgLogo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#22d3ee" />
          <stop offset="1" stopColor="#6366f1" />
        </linearGradient>
      </defs>
      <path d="M16 2.5 28 7v8.5c0 7-5 11.6-12 14-7-2.4-12-7-12-14V7z" fill="none" stroke="url(#tgLogo)" strokeWidth="2" />
      <path d="M11 13c0-2 1.6-3.5 3.5-3.5.7-1.2 2-1.8 3.2-1.5 1.8.4 2.8 2 2.6 3.7 1.3.5 2.2 1.800 2 3.200-.2 1.600-1.500 2.800-3 2.800h-.7V22h-1.600v-3.300h-1V22H15v-3.300c-2.300 0-4-1.700-4-3.800 0-.8.3-1.500 1-1.900z" fill="url(#tgLogo)" opacity=".9" />
    </svg>
  );
}

export function Disclaimer({ children }) {
  return (
    <aside className="tg-disclaimer" role="note">
      <strong>Research prototype.</strong>{' '}
      {children || 'TumorGuard AI is not clinically validated, does not diagnose cancer, and must not be used as an autonomous diagnostic tool. All outputs require review by qualified professionals.'}
    </aside>
  );
}

export function Badge({ tone = 'info', children }) {
  return <span className={`tg-badge tg-badge-${tone}`}>{children}</span>;
}

export function Collapsible({ title, children, defaultOpen = false, badge = null }) {
  const id = useId();
  return (
    <details className="tg-collapsible" open={defaultOpen}>
      <summary aria-controls={id}>
        <span>{title}</span>
        {badge}
      </summary>
      <div id={id} className="tg-collapsible-body">{children}</div>
    </details>
  );
}

export function EmptyState({ title, children, action }) {
  return (
    <div className="tg-empty" role="status">
      <h2>{title}</h2>
      <p>{children}</p>
      {action}
    </div>
  );
}

export function ConfirmDialog({ title, message, confirmLabel = 'Delete', onConfirm, onCancel }) {
  const ref = useRef(null);
  useEffect(() => { ref.current?.focus(); }, []);
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onCancel(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onCancel]);
  return (
    <div className="tg-modal-backdrop">
      <div className="tg-modal" role="alertdialog" aria-modal="true" aria-labelledby="tg-confirm-title" aria-describedby="tg-confirm-msg">
        <h2 id="tg-confirm-title">{title}</h2>
        <p id="tg-confirm-msg">{message}</p>
        <div className="tg-row">
          <button type="button" className="tg-btn tg-btn-ghost" ref={ref} onClick={onCancel}>Cancel</button>
          <button type="button" className="tg-btn tg-btn-danger" onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}
