import { useState } from 'react';
import { filterSortPage } from '../history.js';
import { verdictLabel } from '../verdict.js';
import { ConfirmDialog, EmptyState, Disclaimer } from '../components/common.jsx';

export default function History({ records, onView, onDelete, onClear, navigate }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('newest');
  const [page, setPage] = useState(1);
  const [confirm, setConfirm] = useState(null); // {kind:'one'|'all', id}
  const view = filterSortPage(records, { query, sort, page });
  return (
    <main className="tg-page" id="main">
      <h1>Session history</h1>
      <p className="tg-fine">
        Analyses from this browser tab only. Scores and verdicts are kept in sessionStorage and cleared when the tab closes;
        images and file names are never stored, so a record restored after a reload has no images.
        Nothing is saved on a server.
      </p>
      {records.length === 0 ? (
        <EmptyState title="No analyses in this session" action={<button type="button" className="tg-btn tg-btn-primary" onClick={() => navigate('upload')}>Upload MRI</button>}>
          Completed analyses will be listed here.
        </EmptyState>
      ) : (
        <>
          <div className="tg-toolbar">
            <label className="tg-fine" htmlFor="hist-q">Search</label>
            <input id="hist-q" type="search" value={query} placeholder="Case ID or verdict" onChange={(e) => { setQuery(e.target.value); setPage(1); }} />
            <label className="tg-fine" htmlFor="hist-sort">Sort</label>
            <select id="hist-sort" value={sort} onChange={(e) => { setSort(e.target.value); setPage(1); }}>
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="score_desc">Score, high to low</option>
              <option value="score_asc">Score, low to high</option>
            </select>
            <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" onClick={() => setConfirm({ kind: 'all' })}>Clear history</button>
          </div>
          <div className="tg-table-wrap">
            <table className="tg-table">
              <caption className="tg-visually-hidden">Analyses in this session</caption>
              <thead><tr><th scope="col">Date and time</th><th scope="col">Case ID</th><th scope="col">Slice</th><th scope="col">Verdict</th><th scope="col">Score</th><th scope="col">Actions</th></tr></thead>
              <tbody>
                {view.rows.map((r) => (
                  <tr key={r.id}>
                    <td>{new Date(r.createdAt).toLocaleString()}</td>
                    <td><code>{r.id}</code></td>
                    <td>{r.sliceIndex ?? 'n/a'}</td>
                    <td>{verdictLabel(r.verdict)}</td>
                    <td>{Number.isFinite(r.reliabilityScore) ? r.reliabilityScore.toFixed(4) : 'n/a'}</td>
                    <td>
                      <div className="tg-row">
                        <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" onClick={() => onView(r.id)} aria-label={`View case ${r.id}`}>View</button>
                        <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" onClick={() => setConfirm({ kind: 'one', id: r.id })} aria-label={`Delete case ${r.id}`}>Delete</button>
                      </div>
                    </td>
                  </tr>
                ))}
                {view.rows.length === 0 && <tr><td colSpan={6}>No cases match your search.</td></tr>}
              </tbody>
            </table>
          </div>
          <nav className="tg-pager" aria-label="History pages">
            <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={view.page <= 1} onClick={() => setPage(view.page - 1)}>Previous page</button>
            <span className="tg-fine">Page {view.page} of {view.pages} · {view.total} case{view.total === 1 ? '' : 's'}</span>
            <button type="button" className="tg-btn tg-btn-ghost tg-btn-sm" disabled={view.page >= view.pages} onClick={() => setPage(view.page + 1)}>Next page</button>
          </nav>
        </>
      )}
      {confirm && (
        <ConfirmDialog
          title={confirm.kind === 'all' ? 'Clear all history?' : `Delete case ${confirm.id}?`}
          message={confirm.kind === 'all' ? 'This removes every record from this session. It cannot be undone.' : 'This removes the record from this session. It cannot be undone.'}
          confirmLabel={confirm.kind === 'all' ? 'Clear history' : 'Delete'}
          onCancel={() => setConfirm(null)}
          onConfirm={() => { if (confirm.kind === 'all') onClear(); else onDelete(confirm.id); setConfirm(null); setPage(1); }}
        />
      )}
      <Disclaimer />
    </main>
  );
}
