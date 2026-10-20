import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => { cleanup(); try { sessionStorage.clear(); } catch { /* not available in the node environment */ } });
if (typeof window !== "undefined") window.scrollTo = () => {};
