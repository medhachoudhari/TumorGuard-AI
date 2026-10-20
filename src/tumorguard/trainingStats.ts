// Reference statistics from BraTS 2020 training distribution
// Exactly matches models/training_stats.json from the original research pipeline

export interface TrainingStats {
  image_size: number;
  modality_order: string[];
  normalization_method: string;
  slice_selection: string;
  training_patients: string[];
  model_architecture: string;
  dropout_rate: number;
  familiarity_reference: {
    feature_names: string[];
    mean: number[];
    std: number[];
  };
}

export const TRAINING_STATS: TrainingStats = {
  image_size: 128,
  modality_order: ["flair", "t1ce", "t2"],
  normalization_method: "z_score_per_slice_brain_mask",
  slice_selection: "max_tumor_pixels_from_seg_mask",
  training_patients: ["001", "002", "003", "004"],
  model_architecture: "2D_UNet_MCDropout",
  dropout_rate: 0.3,
  familiarity_reference: {
    feature_names: [
      "flair_mean", "flair_std", "flair_min", "flair_max", "flair_nonzero_frac", "flair_p75",
      "t1ce_mean", "t1ce_std", "t1ce_min", "t1ce_max", "t1ce_nonzero_frac", "t1ce_p75",
      "t2_mean", "t2_std", "t2_min", "t2_max", "t2_nonzero_frac", "t2_p75"
    ],
    mean: [
      -8.537761275739797e-09,
      0.5646560816581458,
      -2.1519684493541718,
      3.320094883441925,
      0.319091796875,
      0.0,
      -8.808129670079268e-09,
      0.5646560536894076,
      -2.730686664581299,
      6.122839450836182,
      0.319091796875,
      0.0,
      -2.3057268316861723e-09,
      0.5646560669724408,
      -1.9431763589382172,
      3.8214547634124756,
      0.319091796875,
      0.0
    ],
    std: [
      1e-06,
      0.015978956303114196,
      0.29789170189438835,
      0.8529435607422867,
      0.017812543333569738,
      1e-06,
      1e-06,
      0.01597895153713397,
      0.15221033506692486,
      1.5466479617820126,
      0.017812543333569738,
      1e-06,
      1e-06,
      0.015978946221582215,
      0.13404069203272523,
      0.5000346051115295,
      0.017812543333569738,
      1e-06
    ]
  }
};
