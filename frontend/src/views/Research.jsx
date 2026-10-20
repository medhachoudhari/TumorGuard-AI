import { useState } from 'react';
import baseline from '../data/historical_baseline.json';
import { downloadText } from '../report.js';
import { Disclaimer } from '../components/common.jsx';

const TABS = [['performance', 'Model performance'], ['method', 'Methodology'], ['limits', 'Limitations'], ['cite', 'Citations']];

const CITATIONS = [
  ['Menze et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE TMI 34(10).', 'https://doi.org/10.1109/TMI.2014.2377694'],
  ['Bakas et al. (2017). Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. Scientific Data 4.', 'https://doi.org/10.1038/sdata.2017.117'],
  ['Ronneberger, Fischer & Brox (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.', 'https://arxiv.org/abs/1505.04597'],
  ['Gal & Ghahramani (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML.', 'https://arxiv.org/abs/1506.02142'],
];

export default function Research() {
  const [tab, setTab] = useState('performance');
  const m = baseline.metrics;
  const rows = [
    ['Validation slices', m.validation_samples], ['Mean Dice', m.mean_dice], ['Median Dice', m.median_dice],
    ['Mean IoU', m.mean_iou], ['Mean precision', m.mean_precision], ['Mean recall', m.mean_recall],
    ['Slices with empty prediction', m.zero_prediction_slices],
  ];
  const onKey = (e) => {
    const i = TABS.findIndex(([k]) => k === tab);
    if (e.key === 'ArrowRight') setTab(TABS[(i + 1) % TABS.length][0]);
    if (e.key === 'ArrowLeft') setTab(TABS[(i + TABS.length - 1) % TABS.length][0]);
  };
  return (
    <main className="tg-page" id="main">
      <h1>Research and validation</h1>
      <div className="tg-tabs" role="tablist" aria-label="Research sections" onKeyDown={onKey}>
        {TABS.map(([k, label]) => (
          <button key={k} type="button" role="tab" id={`tab-${k}`} aria-selected={tab === k} aria-controls={`panel-${k}`} tabIndex={tab === k ? 0 : -1} onClick={() => setTab(k)}>{label}</button>
        ))}
      </div>
      <section role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="tg-card">
        {tab === 'performance' && (
          <>
            <h2>Historical evaluation (reported, not reproduced)</h2>
            <p>{baseline._provenance}</p>
            <table className="tg-table">
              <tbody>{rows.map(([k, v]) => <tr key={k}><th scope="row">{k}</th><td>{Number.isInteger(v) ? v : v.toFixed(4)}</td></tr>)}</tbody>
            </table>
            <p className="tg-fine">Slice-level aggregation over tumour-containing slices only. Patient-level results, error-detection performance (AUROC/AUPRC) and an independent test set have not been produced.</p>
            <button type="button" className="tg-btn tg-btn-secondary" onClick={() => downloadText('tumorguard_historical_baseline.json', `${JSON.stringify(baseline, null, 2)}\n`, 'application/json')}>Download metrics (JSON)</button>
          </>
        )}
        {tab === 'method' && (
          <>
            <h2>Methodology</h2>
            <ul className="tg-list">
              <li>Input: FLAIR, T1CE, T2 axial slices, resized to 128 × 128, z-score normalised per slice within the brain mask (background set to 0).</li>
              <li>Model: 2D U-Net with dropout (p = 0.3), single-channel logits; binary whole-tumour target (BraTS labels 1, 2, 4 merged).</li>
              <li>API prediction: mean of 10 stochastic forward passes (dropout on, BatchNorm in eval mode); mask where mean probability &gt; 0.5. Reported evaluation metrics use one deterministic pass.</li>
              <li>Uncertainty: mean per-pixel variance, mean binary entropy, and the maximum probability difference between the first and last pass.</li>
              <li>Familiarity: distance of 18 image statistics from a 4-patient training reference.</li>
              <li>Reliability score: heuristic weighted combination (0.40 / 0.20 / 0.20 / 0.20); verdict cut-offs 0.70 / 0.50 / 0.30.</li>
              <li>Data: BraTS 2020; training patients 001–030, validation 031–040.</li>
            </ul>
          </>
        )}
        {tab === 'limits' && (
          <>
            <h2>Known limitations</h2>
            <ul className="tg-list">
              <li>Not clinically validated; no external or multicentre validation.</li>
              <li>Validation patients were also used for model selection.</li>
              <li>The familiarity reference uses only four patients.</li>
              <li>Reliability weights and thresholds are uncalibrated heuristics; no evidence yet that they predict segmentation errors.</li>
              <li>Entropy and variance are averaged over the whole slice including background.</li>
              <li>The live slice-selection rule differs from the evaluation protocol; the API mask uses MC averaging while metrics use a deterministic pass.</li>
            </ul>
          </>
        )}
        {tab === 'cite' && (
          <>
            <h2>References</h2>
            <ol className="tg-list">
              {CITATIONS.map(([t, u]) => <li key={u}>{t} <a className="tg-link" href={u} target="_blank" rel="noopener noreferrer">{u.replace('https://', '')}</a></li>)}
            </ol>
          </>
        )}
      </section>
      <Disclaimer />
    </main>
  );
}
