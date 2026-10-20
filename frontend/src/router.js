import { useCallback, useEffect, useState } from 'react';

export const ROUTES = ['home', 'upload', 'analysis', 'results', 'evidence', 'history', 'research', 'about'];

export function parseHash(hash) {
  const name = String(hash || '').replace(/^#\/?/, '').split('?')[0];
  return ROUTES.includes(name) ? name : 'home';
}

export function useRoute() {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => { setRoute(parseHash(window.location.hash)); window.scrollTo?.(0, 0); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const navigate = useCallback((r) => { window.location.hash = `#/${r}`; }, []);
  return [route, navigate];
}
