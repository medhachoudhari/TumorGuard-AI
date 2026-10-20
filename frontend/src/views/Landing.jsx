import baseline from '../data/historical_baseline.json';
import { Disclaimer, Badge } from '../components/common.jsx';

const FEATURES = [
  ['Segmentation', '2D U-Net binary whole-tumour segmentation on FLAIR, T1CE and T2 slices.'],
  ['Uncertainty', 'Ten-pass Monte Carlo Dropout: variance, entropy and pass divergence.'],
  ['Reliability evidence', 'Explicit evidence states and a conservative review recommendation.'],
  ['Human-centred', 'Every output is a prompt for expert review, never a diagnosis.'],
];

export default function Landing({ navigate }) {
  const m = baseline.metrics;
  const learnMore = () => document.getElementById('how-it-works')?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
  return (
    <main className="tg-page tg-landing" id="main">
      <section className="tg-hero">
        <div className="tg-hero-copy">
          <Badge tone="info">Research prototype</Badge>
          <h1>
            Uncertainty-aware brain MRI <span className="tg-grad">tumour segmentation</span> and review prioritisation
          </h1>
          <p className="tg-lead">
            TumorGuard AI studies whether predictive uncertainty and supporting evidence can flag segmentations that
            deserve a second look from a human reviewer.
          </p>
          <div className="tg-row">
            <button type="button" className="tg-btn tg-btn-primary tg-btn-lg" onClick={() => navigate('upload')}>Get started</button>
            <button type="button" className="tg-btn tg-btn-ghost tg-btn-lg" onClick={learnMore}>Learn more</button>
          </div>
        </div>
        <div className="tg-hero-art" role="img" aria-label="Decorative illustration of a neural network. Not an MRI and not a result." />
      </section>

      <section className="tg-stats" aria-label="Historical evaluation summary">
        <dl>
          <div><dt>Mean Dice</dt><dd>{m.mean_dice.toFixed(3)}</dd></div>
          <div><dt>Mean IoU</dt><dd>{m.mean_iou.toFixed(3)}</dd></div>
          <div><dt>Validation slices</dt><dd>{m.validation_samples}</dd></div>
          <div><dt>MC Dropout passes</dt><dd>10</dd></div>
        </dl>
        <p className="tg-fine">
          Historical, reported values from <code>{baseline.source_artifact}</code>; not independently reproduced. The
          validation patients (031–040) were also used for model selection, so this is not an independent test set.
        </p>
      </section>

      <section id="how-it-works" className="tg-section" aria-labelledby="how-h">
        <h2 id="how-h">How it works</h2>
        <ol className="tg-steps">
          <li><strong>Upload</strong> FLAIR, T1CE and T2 NIfTI volumes. Headers are checked before upload.</li>
          <li><strong>Validate</strong> the server checks shapes, alignment and finite values.</li>
          <li><strong>Infer</strong> the U-Net runs ten stochastic passes on the selected slice.</li>
          <li><strong>Review</strong> segmentation, uncertainty and evidence are shown together.</li>
        </ol>
      </section>

      <section className="tg-section" aria-labelledby="feat-h">
        <h2 id="feat-h">What it provides</h2>
        <div className="tg-grid tg-grid-4">
          {FEATURES.map(([t, d]) => (
            <article key={t} className="tg-card"><h3>{t}</h3><p>{d}</p></article>
          ))}
        </div>
      </section>
      <Disclaimer />
    </main>
  );
}
