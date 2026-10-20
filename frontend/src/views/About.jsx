import { Disclaimer } from '../components/common.jsx';

export default function About({ navigate }) {
  return (
    <main className="tg-page" id="main">
      <h1>About TumorGuard AI</h1>
      <p className="tg-lead">A research prototype that investigates whether uncertainty and supporting evidence can help identify brain MRI segmentations that need human review.</p>
      <div className="tg-grid tg-grid-3">
        <article className="tg-card"><h2>Research focus</h2><p>Uncertainty quantification and error identification for binary brain-tumour segmentation on BraTS 2020 FLAIR, T1CE and T2 slices.</p></article>
        <article className="tg-card tg-card-warn"><h2>Not for clinical use</h2><p>Not clinically validated, not approved for patient care, and not a diagnostic tool. Outputs describe model behaviour, not patient health.</p></article>
        <article className="tg-card"><h2>Open science</h2><p>Source code and documentation are on <a className="tg-link" href="https://github.com/medhachoudhari/TumorGuard-AI" target="_blank" rel="noopener noreferrer">GitHub</a>. Limitations are documented in the repository&apos;s defect register.</p></article>
      </div>
      <section className="tg-card">
        <h2>Model and data</h2>
        <ul className="tg-list">
          <li>2D U-Net with Monte Carlo Dropout (PyTorch), served by a local Flask API.</li>
          <li>Dataset: BraTS 2020 training data. Raw data are not distributed with this project.</li>
          <li>Privacy: files are sent only to the server you run; file names and images are not stored by the history feature.</li>
        </ul>
        <div className="tg-row">
          <button type="button" className="tg-btn tg-btn-secondary" onClick={() => navigate('research')}>Research and validation</button>
          <button type="button" className="tg-btn tg-btn-primary" onClick={() => navigate('upload')}>Start an analysis</button>
        </div>
      </section>
      <Disclaimer />
    </main>
  );
}
