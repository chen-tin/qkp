# Data-driven decoding of quantum error correcting codes using graph neural networks

[2025 publication](https://doi.org/10.1103/PhysRevResearch.7.023181) · transferable; surface and repetition codes

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| d3_r3_curve | 0.0006 | fraction | Figure 3; arXiv v2 PDF page 7 |
| d5_r3_curve | 4e-05 | fraction | Figure 3; arXiv v2 PDF page 7 |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
The author repository includes trained checkpoints for each distance and duration. Start with d=3, rounds=3, memory-Z and Appendix B noise. Include empty-syndrome shots in the denominator even though training omits those graphs. Figure 3 uses 10 million shots for d<7 and 100 million for larger distances. Benchmark latency separately; batched GPU throughput is not real-time latency.

Baseline: MWPM and belief-matching, identical Appendix B circuit and memory-Z observable
## Commands
Run from this folder (quote its path because it contains spaces):
```sh
python3 reproduce.py check
python3 reproduce.py plan
python3 reproduce.py fetch --kind paper
python3 reproduce.py fetch --kind code
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
Production commands above are explicit opt-ins. `train` runs the released training example; unresolved gaps below prevent calling it a full paper reproduction. `smoke` is infrastructure validation only. `replay` recomputes author data, without retraining.

## Known gaps
- Full checkpoint evaluation adapter needs PyTorch Geometric extensions and exact graph preprocessing.
- Reference values here are approximate readings of plotted markers; digitize the vector figure or obtain raw author data for tighter matching.
- Do not use generic Stim uniform noise in place of Appendix B.

## Author source
[LangeMoritz/GNN_decoder](https://github.com/LangeMoritz/GNN_decoder) pinned at `15d8443bedf7862ef72ec1b5239ecd01163eb8d6`. This is a reviewed current snapshot, not a claim that it is the paper-era commit. `fetch` leaves existing checkouts intact.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.
