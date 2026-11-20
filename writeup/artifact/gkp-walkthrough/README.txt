GKP illustrated walkthrough

Open viewer.html in a modern browser. It works offline with embedded Matplotlib figures.
15 standalone PNG/SVG figures and 50 round-by-round PNG frames are included.
The Noise playground adds live, seeded JavaScript visualizations with parameter sliders, paired reference traces, and CSV export.
Figures distinguish actual encoder output, analytic curves, and illustrative trajectories.
Illustrations are teaching examples; decoder heatmaps are a separate exploratory development sweep, not a replacement for the benchmark suite.
Seed: 20261117; selected trial (zero-based): 0; highlighted round: 34.
The first trial/round with nearest-lattice error and Bayes success was selected from 32 trials.

Regenerate in the qkp repository:
.venv/bin/python -m writeup.artifact.build_gkp_walkthrough

Decoder heatmaps: three pairwise stationary-noise sweeps with fixed checkpoints, paired trials, and 95% pointwise Wilson intervals.
Run: .venv/bin/python -m experiments.noise_heatmaps
Refresh the offline page without recomputing illustrations: .venv/bin/python -m writeup.artifact.refresh_gkp_viewer
Train/resume the 36-model LSTM width-depth sweep and populate its heatmaps: .venv/bin/python -m experiments.lstm_size_sweep

The Nature paper guide is included in viewer.html and in paper.html, with eight primary references and paper-references.bib.
