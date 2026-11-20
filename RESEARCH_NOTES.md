# GKP decoding and simulator research notes

Last consolidated: **2026-11-20**, America/Los_Angeles. This is the canonical notebook for a future paper, maintained under [AGENTS.md](AGENTS.md).

This first consolidation is retrospective: it summarizes the available code, saved artifacts, protocols and historical findings. It does not assert that all historical experiments were independently rerun today. Dates below follow the source records; run timestamps ending in Z or +00:00 are UTC. New work should be logged as it happens.

The project directory is named qkp; the code studied here is the **Gottesman–Kitaev–Preskill (GKP)** oscillator encoding. The [original user-supplied notes](docs/user_research_notes_2026-11-19.md) preserve the supplied wording and unfinished ideas, with whitespace normalized. Corrections and evidence limits appear here separately.

## Research question and current status

Question: how do syndrome-history decoders fail under correlated displacement noise, changes in noise statistics, and finite-energy measurement dynamics? Compare nearest-lattice, Bayesian and learned sequence methods, while measuring simulator error separately from decoding error.

| Work item | Status as supported by available records | Main evidence |
| --- | --- | --- |
| Classical stationary comparison, E0–E8 | Historical completed baseline; later studies use separate budgets and splits | [RESULTS.md](RESULTS.md), [findings](docs/findings.md) |
| Stationary decoder heatmaps | Complete: 20 decoders × 61 settings | [saved summary](results/noise_heatmaps/e940f6fb63229b90/summary.json) |
| LSTM width × depth × seed sweep | Complete: 36 models, 2,196 evaluations, audited | [study](docs/lstm_size_sweep.md), [completion](results/lstm_size_sweep/full/9aab4963bc37dcea/completion.json) |
| Convy 2022 target | Three diagnostic LSTM seeds complete; published-control mismatch unresolved; no successful paper reproduction | [control audit](docs/convy2022_control_audit.md), [status](results/reproduce_convy2022/anchor/bfcba380fc8e8c55/status.json) |
| Baireuther 2018 Figure 3 target | Completed registered run; measured interval excludes published target | [protocol](docs/baireuther2018_reproduction.md), [result](results/reproduce_baireuther2018/figure3/aaacc7ab3e3a280d/result.json) |
| Wan 2020 | Tracking/posterior components pass; scoring audit completed; full Figure 3 reproduction incomplete | [components](docs/wan2020_reproduction.md), [audit](docs/wan2020_fidelity_audit.md) |
| Other research-backed architectures | Eleven reproduction workspaces; selected software smokes and author-data reanalysis; not eleven completed reproductions | [catalog](misc%20methods/README.md), [validation](misc%20methods/VALIDATION.md) |
| Finite-energy decoder reference | Bounded two-cycle comparison saved; validity depends on point/controller; later eight-cycle intervention covers nearest-lattice at nominal noise only | [reference](docs/qutip_reference.md), [follow-ups](docs/qutip_reference_next_stage.md) |
| Explicit QuTiP versus local measurement | Complete within finite-operator, single-extraction scope | [results](docs/qutip_engine_benchmark_results.md) |
| Smooth-resource/sampler decision | Complete; adopted for new reference runs; nominal eight-cycle checks pass at 4,096/5,120 levels | [decision](docs/smooth_ancilla_results.md), [HTML review](results/smooth_ancilla_experiment/main/06c34206dd1a1b00/review/index.html) |
| Simulation-quality infrastructure | Implemented and verified; production subsequently launched on EC2 | [runbook](docs/simulation_quality.md), [verification](docs/simulation_quality_validation.md) |
| Production EC2 execution | Numerical baseline passed; physics 13/20 and noise 13/14 observations saved, latest refinements pending; coverage/rare queued; finite-gate controls fail population gates | [EC2 run record](docs/ec2_production.md), [live status](results/ec2_runs/production_20261119/metrics/status.json) |
| E9–E13 and RL extensions | Infrastructure and some historical runs/checkpoints exist, but completion varies; requires per-artifact audit before a paper-wide completion claim | [historical findings](docs/findings.md), [nonstationary](docs/nonstationary.md), [learned](docs/learned.md), [hardware](docs/hardware.md) |

Older overview files contain launch-time phrases such as “not run” or “running.” The dated entries and run artifacts below supersede those phrases only for the specific completed targets. A later LSTM size sweep does not automatically complete all of E13.

## Prior work: reinforcement learning

These are literature references, not local reproduction results. Bibliographic identity/links were checked on 2026-11-19 where noted; a checked identity does not mean every numerical claim or implementation detail has been audited.

1. T. Fösel, P. Tighineanu, T. Weiss, and F. Marquardt, “Reinforcement Learning with Neural Networks for Quantum Feedback,” Physical Review X **8**, 031084 (2018). [Primary paper](https://doi.org/10.1103/PhysRevX.8.031084). Motivation for learning feedback strategies.
2. M. Bukov, A. G. R. Day, D. Sels, P. Weinberg, A. Polkovnikov, and P. Mehta, “Reinforcement Learning in Different Phases of Quantum Control,” Physical Review X **8**, 031086 (2018). [Primary paper](https://doi.org/10.1103/PhysRevX.8.031086). Quantum-control precedent, with a different task from GKP decoding.
3. V. V. Sivak, A. Eickbusch, H. Liu, B. Royer, I. Tsioutsios, and M. H. Devoret, “Model-Free Quantum Control with Reinforcement Learning,” Physical Review X **12**, 011059 (2022). [Primary paper](https://doi.org/10.1103/PhysRevX.12.011059). Model-free circuit-parameter learning.
4. F. Metz and M. Bukov, “Self-correcting quantum many-body control using reinforcement learning with tensor networks,” Nature Machine Intelligence **5**, 780–791 (2023). [Primary paper](https://doi.org/10.1038/s42256-023-00687-5). Many-body control and tensor-network setting; transfer to this decoder is untested.
5. M. Bukov and F. Marquardt, “Reinforcement Learning for Quantum Technology,” arXiv:2601.18953v1 (2026). [Versioned review](https://arxiv.org/abs/2601.18953v1). Review and source of further references; not an original local performance result.

## Prior work: experimental RL applications

1. L. Ding, M. Hays, Y. Sung, B. Kannan, J. An, A. Di Paolo, A. H. Karamlou, T. M. Hazard, K. Azar, D. K. Kim, et al., “High-Fidelity, Frequency-Flexible Two-Qubit Fluxonium Gates with a Transmon Coupler,” Physical Review X **13**, 031035 (2023). [Primary paper](https://doi.org/10.1103/PhysRevX.13.031035). The paper explicitly uses model-free RL pulse optimization and reports mean CZ fidelity 99.922% ± 0.009%; this gate metric is not a GKP logical-memory metric.
2. Y. Baum, M. Amico, S. Howell, M. Hush, M. Liuzzi, P. Mundada, T. Merkh, A. R. R. Carvalho, and M. J. Biercuk, “Experimental Deep Reinforcement Learning for Error-Robust Gate-Set Design on a Superconducting Quantum Computer,” PRX Quantum **2**, 040324 (2021). [Primary paper](https://doi.org/10.1103/PRXQuantum.2.040324). Experimental error-robust gate design; local reproduction not run.
3. K. Reuer, J. Landgraf, T. Fösel, J. O’Sullivan, L. Beltrán, A. Akin, G. J. Norris, A. Remm, M. Kerschbaum, J.-C. Besse, et al., “Realizing a deep reinforcement learning agent for real-time quantum feedback,” Nature Communications **14**, 7138 (2023). [Primary paper](https://doi.org/10.1038/s41467-023-42901-3). FPGA feedback for superconducting-qubit initialization; local reproduction not run.

## QC notes and GKP encoding

The Knill–Laflamme condition for exactly correcting an error set is

    P E_a† E_b P = c_ab P,

where P projects onto the code subspace. This specifies correctability for the stated error set and code. It does not certify a finite-energy approximate encoding, a finite basis, or a particular learned recovery. [John Wright, lecture 4, scribed by Shilun Li (2024)](https://people.eecs.berkeley.edu/~jswright/quantumcodingtheory24/scribe%20notes/lecture04.pdf).

Conventions: [q,p] = i, hbar = 1, and L = sqrt(pi). Ideal logical position peaks are q = 2nL for zero and q = (2n+1)L for one. A displacement of L changes logical position parity; 2L is a stabilizer displacement. The scalar nearest-lattice rule corrects shifts with magnitude below L/2 in the ideal noiseless-measurement case. Finite-energy envelopes and noisy extraction require additional analysis.

**Correction to the supplied draft:** peak positions use multiples of sqrt(pi), not division by sqrt(pi). The “even/odd k sqrt(pi)” statement in the encoding section was the correct convention.

Core sources:

- D. Gottesman, A. Kitaev, and J. Preskill, “Encoding a qubit in an oscillator,” Physical Review A **64**, 012310 (2001). [DOI](https://doi.org/10.1103/PhysRevA.64.012310), [arXiv:quant-ph/0008040](https://arxiv.org/abs/quant-ph/0008040).
- A. L. Grimsmo and S. Puri, “Quantum Error Correction with the Gottesman–Kitaev–Preskill Code,” PRX Quantum **2**, 020101 (2021). [DOI](https://doi.org/10.1103/PRXQuantum.2.020101). Encoding, Gaussian displacements and extraction context.
- K. Fukui, A. Tomita, A. Okamoto, and K. Fujii, “High-Threshold Fault-Tolerant Quantum Computation with Analog Quantum Error Correction,” Physical Review X **8**, 021054 (2018). [DOI](https://doi.org/10.1103/PhysRevX.8.021054). Local validation notes use Eq. (22) at zero postselection cutoff; equation numbering should be checked against the final version before manuscript submission.
- K. H. Wan, A. Neville, and S. Kolthammer, “Memory-assisted decoder for approximate Gottesman–Kitaev–Preskill codes,” Physical Review Research **2**, 043280 (2020). [DOI](https://doi.org/10.1103/PhysRevResearch.2.043280), [user-linked Imperial copy](https://spiral.imperial.ac.uk/server/api/core/bitstreams/848a611d-955e-4b4f-89ae-1a7539925da2/content). This identifies the previously unnamed simulation reference. Our classical AR(1) model is not the complete finite-energy protocol from this paper.
- T. Matsuura, H. Yamasaki, and M. Koashi, “Equivalence of approximate Gottesman-Kitaev-Preskill codes,” Physical Review A **102**, 032408 (2020). [DOI](https://doi.org/10.1103/PhysRevA.102.032408). Approximate-state representation.

## Noise: distribution, units, and parameter choices

### Classical displacement/parity simulator

Actual module: [simulator/noise.py](simulator/noise.py). This is a scalar q-displacement model with a separate measurement-noise array, not a full two-quadrature quantum-state simulation.

    epsilon_1 ~ Normal(0, sigma_q^2)
    epsilon_t = rho epsilon_(t-1) + eta_t
    eta_t ~ Normal(0, sigma_q^2 (1-rho^2))
    nu_t ~ Normal(0, sigma_m^2), independently over rounds

Displacement variance remains sigma_q² for every round because initialization is stationary. The innovation standard deviation is sigma_q sqrt(1-rho²). Thus increasing rho increases predictability without changing the marginal displacement variance. rho=0 gives independent displacements; the implementation requires -1 < rho < 1. Measurement noise is independent of the displacement stream in this model.

| Parameter | Meaning | Nominal final setting for the stationary baseline and size sweep |
| --- | --- | ---: |
| sigma_q/L | Displacement standard deviation, not variance | 0.20 |
| sigma_m/L | Additive readout standard deviation in the classical simulator | 0.05 |
| rho | Lag-one correlation of per-round displacement increments | 0.90 |
| T | Sequence length for stationary heatmaps/size sweep | 50 rounds |

The numerical scales and correlations are **controlled stress-test choices**, not values all inferred from GKP hardware or prescribed by one paper. Historical hardware-scale calculations use sigma_q/L≈0.072 and a separately calibrated effective readout scale; that calibration does not measure rho. See [benchmarks, especially B and H](docs/benchmarks.md). Calibration to one performance value is not independent hardware validation.

### Gradual noise drift

[simulator/nonstationary.py](simulator/nonstationary.py) stores per-trial/per-round sigma_q, sigma_m, rho, kick and event time. The displacement recurrence rescales the prior increment when sigma changes:

    epsilon_t = rho_t (sigma_t/sigma_(t-1)) epsilon_(t-1)
                + sigma_t sqrt(1-rho_t²) z_t.

Schedules include sinusoidal changes in sigma_q or rho, ramps, and a clipped sum of log-spaced OU processes used as a 1/f-like envelope. Amplitude, period, phase, range, target and clipping are part of the configuration, not implicit defaults to omit from a paper. See [model/protocol](docs/nonstationary.md) and [historical stress-test findings](docs/findings.md).

### Unexpected noise changes

Implemented families: unannounced steps, two-level telegraph switching and one-round extra displacement bursts. Event time, transition levels, switch probability and burst standard deviation must be recorded. These are different interventions and should not be combined into a single “nonstationary” score without specifying weights.

### Unseen noise conditions and robustness

Hold out shapes, parameter ranges, and time scales separately. The stationary heatmap uses:

| Axis | Values |
| --- | --- |
| sigma_q/L | 0.08, 0.14, 0.20, 0.26, 0.32 |
| sigma_m/L | 0.04, 0.05, 0.08, 0.12, 0.16 |
| rho | 0, 0.30, 0.60, 0.90, 0.97 |

Three intersecting 5×5 planes hold one parameter nominal at a time: q/rho at sigma_m/L=.05, q/m at rho=.90, and rho/m at sigma_q/L=.20. Their union has **61 settings**, not all 125 combinations. These planes expose some pairwise interactions but leave general three-way interactions and untested distributions open. There is no negative-correlation, loss, dephasing or arbitrary non-Gaussian robustness claim from this sweep.

## Simulation and validation

### Classical update: simulator/sim.py

For residual x, sampled displacement epsilon, and readout noise nu:

1. x_pre = x_previous + epsilon.
2. Syndrome s = wrap(x_pre + nu) into a lattice cell of width L.
3. Decoder maps available syndrome history to correction c.
4. Residual x = x_pre - c.
5. After an ideal final lattice correction, a logical error is odd round(x/L).

For parity-only corrections c=s+jL, syndrome histories can be shared across decoders. That provides paired comparisons. Arbitrary continuous corrections require the closed-loop simulator and a separately specified policy.

[Validation derivations and results](docs/sim_validation.md) compare one-round odd-cell Gaussian probabilities, ideal-readout multiround parity, and the nearly frozen-noise limit. The one-round odd-cell sum agrees with the cited GKP/Fukui conventions. At physical q-standard-deviation .555, historical simulation gives .11076 versus .11031 from the exact sum, consistent within sampling error. Multi-round independent-parity formulas are exact only under their independence assumptions; readout-noise differences induce adjacent correlations, and rho≠0 requires further treatment. Older prose calling Gaussian tail events “almost sure” should be read as negligible-probability approximations, not exact bounded support.

### Finite-energy reference and QuTiP

[quantum/qutip_reference.py](quantum/qutip_reference.py) contains the historical local Fock-space machinery and bin resource; using QuTiP for operators/state preparation does not make that bin lookup a built-in QuTiP measurement algorithm. [quantum/measurement_benchmark.py](quantum/measurement_benchmark.py) supplies the independent explicit two-mode circuit comparison. Shared preparation/scoring is a stated limit of independence.

The finite-energy reference applies q then p displacement/extraction with fresh ancillary resources and explicit measurement back-action. It fixes encoding width, Fock cutoff, homodyne representation, logical extraction and shortest lattice-equivalent feedback. Equal parameter names do not make this the identical classical additive-readout channel.

The current smooth resource is in [quantum/continuous_ancilla.py](quantum/continuous_ancilla.py): continuous Mehler evaluation plus a matching positive-Gaussian-mixture inverse-CDF sampler. The sampler, optimized evaluation and reference weighting are our derived implementation. The physical extraction framework has literature support; this is not a claim to rerun a published benchmark with identical code.

## Decoders and learning

### 1. Nearest lattice

Memoryless c=s. It uses the current syndrome alone and has the ideal L/2 decision boundary. It is the baseline for gain and paired comparisons; its finite-energy behavior also depends on extraction and feedback conventions.

### 2. Bayesian methods

The grid filter tracks displacement memory and cumulative logical parity under an assumed stochastic model. It is a numerical approximation whose grid convergence must be checked. Matched Bayes knows the true model parameters in these synthetic studies and is an informed reference; it is not a parameter-free deployable decoder. Variants include fixed nominal, an eight-model full-history mixture, regime-switching, oracle schedules, IMM and windowed weighting. See [historical findings](docs/findings.md), [nonstationary protocols](docs/nonstationary.md), and [code](decoders/bayesian/grid_filter.py).

Historical E1 nominal P_L(50): matched Bayes≈8.5e-4, broad Bayes≈9.7e-4, nearest lattice≈.242, and original nominal LSTM seed mean≈.0066. These use the historical E1 budget (60,000 trials) and original checkpoints, not the later 1,024-trial size sweep. The broad “Bayes is best” wording in older reports is restricted to those tested methods/settings and should not be extrapolated to all architectures or physical noise.

Wan 2020 supports the motivation for multi-round Bayesian GKP estimation. It does not establish that our AR(1) discretization, selected grid or noise values reproduce that physical protocol.

### 3. LSTM

Supervised sequence learning on synthetic simulator trajectories. Inputs at time t are s/L, sin(2pi s/L), and cos(2pi s/L). Five causal heads predict parity over windows 1, 2, 3, 5 and the full prefix. Binary cross-entropy is averaged over sequences, rounds and heads. Hidden state resets for each sequence; inference receives no future syndrome or hidden-noise labels.

The completed size sweep uses 200,000 training and 20,000 validation sequences per seed; widths 32/64/128/256 × depths 1/2/3 × seeds 0/1/2. Within a seed, architectures share sequences and minibatch permutations. Float32, AdamW lr=.002, weight decay=.0001, batch=2,048, gradient clipping=1, cosine schedule, maximum 40 epochs and validation patience 5 with minimum improvement 1e-5. Restore validation-best weights and select the head by validation mean logical error. No evaluation heatmap selects a checkpoint or head.

All models train at the nominal point. Their 61-setting evaluation uses 1,024 shared trajectories per cell on development streams, root seed 20261109. Separate seed models do not turn a common evaluation set into independent new noise trials. Training choices and periodic multi-head features are project choices. This measures capacity under one budget and training distribution, not the best possible LSTM.

Method precedent: P. Baireuther, T. E. O’Brien, B. Tarasinski, and C. W. J. Beenakker, “Machine-learning-assisted correction of correlated qubit errors in a topological code,” Quantum **2**, 48 (2018), [paper](https://quantum-journal.org/papers/q-2018-01-29-48/), Appendix A. Their surface-code architecture/protocol differs. Optimizer reference: I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).

### 4. Other architectures and encoding methods

The [misc methods catalog](misc%20methods/README.md) has paper-named folders for ResGRU, GKP CNN, TCN, AlphaQubit, Mamba/state-space models, GNN, neural belief propagation, non-Markovian feedback, learned approximate GKP encodings, analog bosonic LDPC, and surface-GKP joint decoding. Each includes published quantitative targets, conditions, provenance and reproduction infrastructure. Missing implementations/data and dependency conflicts are explicit.

Only some software checks have executed: ResGRU/TCN/NBP4 smokes and Mamba released-data reanalysis. For Mamba's d=3, r=28, p=.002 simulated author data, four 500,000-shot sets were rescored: with injected decoder noise Mamba/Transformer errors=29,761/31,568; without, 29,529/31,361. This checks arithmetic on released observations, not newly trained performance or a hardware per-round claim. See [validation](misc%20methods/VALIDATION.md).

No publication backing was established for our proposed Bayesian–neural residual hybrid as an already benchmarked GKP method. Keep it as a hypothesis. Transfer from surface/LDPC/qubit tasks to our GKP setting is a separate experiment.

## Metrics and statistical rules

| Metric | Definition and direction | Interpretation / uncertainty |
| --- | --- | --- |
| P_L(t) | Fraction with incorrect cumulative logical parity at round t; lower is better | Report errors/trials, horizon and pointwise Wilson intervals; compare paired decoder outcomes on common trajectories |
| Time-averaged logical error | Mean P_L(t) over the specified rounds; lower is better | Different from terminal P_L; rounds within a trajectory are dependent |
| epsilon, logical error per round | Fit decay of 1−2P_L(t) before saturation, or the explicitly specified paper fit | Not raw flips/round and not terminal errors divided by rounds; fit window/model/uncertainty required |
| Exact-final-state classification | Fraction exactly matching the full final basis state | Convy task-specific metric; differs from majority recovery and time-average accuracy |
| Logical memory fidelity | Fidelity of extracted logical output with the intended qubit, with stated input average | Finite-energy performance; retain cycle-zero encoding offset; not implementation agreement A |
| Conditional-state agreement A | Integral p_Q(y) times squared overlap of QuTiP and model conditional pure states | Higher is better; our reference weighting; report minimum over tested cases; no universal channel/hardware guarantee |
| Outcome total variation | One half integral of absolute density difference | Lower is better; outcome distribution discrepancy separate from conditional-state discrepancy |
| Norm error / boundary mass | State normalization error / mass in top 10% of the Fock basis | Numerical diagnostics, including worst individual trajectory and substeps; not decoder error rates |
| Mean and maximum photon number | Expectation of number operator, aggregated as specified | Tracks heating and rare tails; norm preservation alone cannot certify truncation |
| Inverse-CDF residual / empirical CDF distance | Sampler quantile error / distance from target CDF | Algorithmic versus sampling diagnostics; separate deterministic tolerance from DKW–Massart statistical bound |

Squared fidelity convention: R. Jozsa, “Fidelity for Mixed Quantum States,” Journal of Modern Optics **41**, 2315–2323 (1994), [DOI](https://doi.org/10.1080/09500349414552171). For pure states, F=|overlap|². The outcome weighting and numerical accuracy tolerance in our experiment are ours.

Empirical CDF reference: P. Massart, “The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality,” Annals of Probability **18**, 1269–1283 (1990), [DOI](https://doi.org/10.1214/aop/1176990746). The run specifies sample size and multiplicity adjustment.

Report training-seed spread separately from sampling uncertainty. State pointwise versus simultaneous intervals. An equivalence claim requires the full interval inside the registered equivalence band; an interval containing zero alone does not prove equivalence. Numerical refinement sensitivity is not automatically a rigorous error bound.

## RL notes, policies, and multi-agent ideas

User's starting reference: V. Sivak et al., “Reinforcement Learning Control of Quantum Error Correction,” [arXiv:2511.08493v3](https://arxiv.org/abs/2511.08493v3). The abstract reports continuous calibration from detection events, 3.5-fold surface-code stability against injected drift, and task-specific surface/color logical-error values. This is a hardware surface/color-code control result, not our GKP decoder result. The supplied v3 is preserved; arXiv lists v4 separately, so future numerical comparisons must pin a version.

Local RL infrastructure includes a closed-loop environment, MLP DQN, replay buffer and parity/fine-action policies. See [continuous corrections/RL](docs/continuous_rl.md), [rl/env.py](rl/env.py), and [rl/dqn.py](rl/dqn.py). Candidate reward choices are dense wrong-parity penalty, terminal failure and parity-change penalty. The latter is not a valid long-run logical-error metric after saturation. Existing infrastructure or a training checkpoint does not establish a completed RL performance study; current paper-ready RL completion remains unaudited.

Multi-agent starting reference: M. Guatto, F. Preti, M. Schilling, T. Calarco, F. A. Cárdenas-López, and F. Motzoi, “Real-time adaptive quantum error correction by model-free multi-agent learning,” [arXiv:2509.03974v2](https://arxiv.org/abs/2509.03974v2). Version checked 2026-11-19. Offline agents construct encoding, syndrome extraction and recovery without a prescribed code/circuit ansatz; online BRAVE retunes a smaller variational parameterization. The abstract reports roughly 18-fold/3-fold logical-infidelity reductions for its qubit/qutrit tasks under specified sampling/drift regimes. These are author-reported results, not local reproductions or GKP transfer evidence.

The user's cooperative versus adversarial distinction and [Hugging Face MARL introduction](https://huggingface.co/learn/deep-rl-course/en/unit7/introduction-to-marl) remain useful background. A cooperative controller/decoder and an adversarial noise generator would change the learning problem and evaluation distribution; no such local multi-agent result is claimed. Policies, observations, actions, reward, horizon and train/test opponent distributions remain design questions.

## Stack and artifacts

- Local Python environment: .venv; use its interpreter. Exact software/device versions belong to each run manifest, not a single timeless notebook version string.
- NumPy/SciPy for sampling, filtering, quadrature and inference; PyTorch for learned decoders; QuTiP for oscillator operators and independent circuit controls; Matplotlib for scientific figures.
- HTML walkthrough: [viewer](writeup/artifact/gkp-walkthrough/viewer.html), [generator](writeup/artifact/build_gkp_walkthrough.py), [metrics dashboard](writeup/artifact/metrics/viewer.html), [reproduction report](writeup/artifact/gkp-walkthrough/reproductions.html). Saved-data reports do not launch training by being opened.
- Existing manuscript sources: [writeup/report.tex](writeup/report.tex), supporting writeup sections and methods. These predate parts of this notebook and require a claim/citation audit before reuse.
- [Harbor](https://www.harborframework.com/) was supplied as a candidate framework. No adoption or integration was verified in this consolidation.
- Large local arrays/environments/results may be ignored or untracked. Preserve run hashes and a durable artifact archive before submission; the existence of local files alone does not ensure a complete published reproduction package.

## Dated work log

### 2026-11-14–16 — Historical classical comparisons and extensions

**Status:** retrospective summary of [RESULTS.md](RESULTS.md), [findings](docs/findings.md), and supporting docs; not freshly rerun.

**Work:** E0–E8 simulator/filter validation, nominal comparisons, correlation/displacement/readout sweeps, mismatch, regime switching, timing and diagnostics. Additional work built drift, hardware-scale, binary-readout, continuous-action and learned-decoder infrastructure. Later findings record E5b mismatch and E14/E15 measurements; some older overview status labels are stale.

**Evidence/result:** nominal historical numbers are listed under Bayesian methods. E5b finds overconfident wrong models dangerous; E14 explores bits per round; E15 finds fine corrections largely information-neutral in the classical shift model. Consult per-experiment files for budgets and exact conditions rather than combining these results as one dataset.

**Correction/decision:** the 2026-11-15 metric audit established that raw parity-change frequency becomes misleading near saturation. Retain P_L(t), pre-saturation decay fits and paired horizon comparisons. Some throughput conclusions differ by hardware/batching in historical notes; no universal LSTM speed advantage is carried forward.

### 2026-11-17–18 — Noise controls, parameter heatmaps, HTML and data coverage

**Status:** complete stationary development grid. Work added an interactive noise playground, decoder heatmaps, a paper guide, source links and offline packaging. Initial missing LSTM cells motivated explicit measured-versus-pending coverage displays.

**Configuration/data:** 61 unique stationary settings listed above, 50 rounds, 1,024 trials/cell, 20 original decoders; shared development noise streams, seed root 20261109. Run e940f6fb63229b90 reports 61/61 completed settings. No E1–E15 raw results are overwritten.

**Metric:** P_L(t), cell counts/intervals and paired comparisons; grid means are equally weighted descriptive averages, not an operating distribution. Fast/reference checks are saved with the run.

**Evidence:** [summary](results/noise_heatmaps/e940f6fb63229b90/summary.json), [plan](results/noise_heatmaps/e940f6fb63229b90/plan.json), [validation](results/noise_heatmaps/e940f6fb63229b90/validation.json), [runner](experiments/noise_heatmaps.py), [viewer](writeup/artifact/gkp-walkthrough/viewer.html).

**Decision/limits:** retain as an exploratory failure map; it is not complete coverage of all noise combinations or hardware-calibrated robustness. A zero count at 1,024 trials is not a zero error probability.

### 2026-11-18 — Completed LSTM size sweep

**Question:** how do width and depth change performance at fixed training distribution/budget?

**Status/configuration:** run 9aab4963bc37dcea, all 36 checkpoints and 2,196 model/setting evaluations complete. Thirty-two models reached 40 epochs; four met registered early stopping. Training took 259.4 minutes on Apple MPS, excluding evaluation/packaging. Full protocol is under LSTM above and in [study](docs/lstm_size_sweep.md).

**Metric/results:** one-layer nominal mean P_L(50) falls from 7.715% at width32 to .488% at width256. Lowest nominal architecture mean is .391% at width256/depth3; its per-seed errors are 3, 3 and 6 out of 1,024. Lowest equally weighted grid mean is 27.289% at width256/depth1, versus 27.515% at width256/depth3. These exploratory differences do not establish a universal winner.

**Failure examples:** width256/depth3 nominal mean .391% rises to 47.884% when rho changes from .9 to 0 with other parameters fixed; sigma_m/L=.16 at rho=.9 gives 51.432%. These include intrinsic difficulty and training mismatch; matched Bayes also performs poorly in those hard cells.

**Checks/evidence:** 72 fast/reference checks agree exactly; the saved audit verifies all counts/checkpoints and preserves original comparisons. [Audit](results/lstm_size_sweep/full/9aab4963bc37dcea/audit.json), [seed results](results/lstm_size_sweep/full/9aab4963bc37dcea/size_comparison.csv), [architecture means](results/lstm_size_sweep/full/9aab4963bc37dcea/architecture_comparison.csv).

**Decision:** retain completed capacity baseline. Next robustness study should vary training-noise coverage at a fixed architecture and evaluate on separately held-out trajectories/settings.

### 2026-11-18–19 — Published reproduction targets

**Purpose:** reproduce prior methods in their original tasks before interpreting GKP transfer. [Registered overview](docs/published_reproductions.md) contains launch-time status; the completed artifacts below supersede it for these targets.

| Target | Our measured outcome | Published comparison and interpretation |
| --- | --- | --- |
| Convy Figure2 SchemeA, gamma=.04/us | Bayes 25,194/30,000=.8398, Wilson95 [.835606,.843907]; LSTM seeds .834833/.836100/.834433 | Digitized Bayes≈.8868 and LSTM≈.8835667; control discrepancy unresolved; diagnostic methods run, not successful numerical reproduction |
| Baireuther Figure3 | epsilon=.0021761764/cycle, trajectory-bootstrap95 [.0021352154,.0022192424], 50,000 test histories, validation-selected seed0 | Published .00209 lies outside interval even with registered ±.000005 rounding allowance; completed run fails the registered numerical-match criterion |
| Wan AppendixD tracking | 356/20,000,000=1.78e-5, Wilson95 [1.60445e-5,1.97475e-5]; independent quadrature1.8026651e-5 | Component scale and independent calculations agree; does not reproduce logical lifetime |
| Wan Gaussian posterior | Recursion and matrix variance agree within5.5e-20; 200,000 trajectories,200 rounds | Local Gaussian component check, excluding tracking/truncation failures |
| Wan Figure3 scoring audit | At200 rounds, no-QEC plus-state fidelity .91163403 versus all-pure-state minimum .90193657;41 plus-state points within .0003 of digitized curve | Published “worst” convention unresolved; full corrected channels not implemented; do not choose a score merely to match the graph |

**Convy provenance:** [protocol](docs/convy2022_reproduction.md), [control audit](docs/convy2022_control_audit.md), [seed0](results/reproduce_convy2022/anchor/bfcba380fc8e8c55/lstm_s0_metrics.json), [seed1](results/reproduce_convy2022/anchor/bfcba380fc8e8c55/lstm_s1_metrics.json), [seed2](results/reproduce_convy2022/anchor/bfcba380fc8e8c55/lstm_s2_metrics.json). Three seeds share the same test data. Independent author-filter recurrence checks agree with our control; original neural Colab is unavailable. Do not substitute time-average or majority recovery for the registered exact-final-state metric.

**Baireuther provenance:** [protocol](docs/baireuther2018_reproduction.md), [source-equivalence validation](reproductions/baireuther2018/validation.json), [completed result](results/reproduce_baireuther2018/figure3/aaacc7ab3e3a280d/result.json), completion timestamp2026-11-19T14:55:05+00:00. Four million training histories, three training seeds, author stopping rule and PyTorch reconstruction details are preserved. The mismatch is a result for this reconstruction; it does not disprove the published work.

**Wan provenance:** [component run](results/reproduce_wan2020/components/91b306a688bce429/), [fidelity audit](results/reproduce_wan2020/fidelity_audit/ea2e740f9366feba/), [protocol](docs/wan2020_reproduction.md), [scoring protocol](docs/wan2020_fidelity_audit.md). Published journal and arXiv versions differ materially; pin the journal target and never substitute preprint bounds silently.

**Decision:** report both successes and unresolved/failed targets. Do not label all three papers reproduced. Future adjustments require new protocols/run identities, not test-set tuning of the completed run.

### 2026-11-18 — Other-method reproduction infrastructure

**Status:** eleven paper workspaces created, each with quantitative published targets, metric contracts, dependency notes, retrieval/provenance and evidence scorer. Individual missing model runners are documented. [Catalog](misc%20methods/README.md), [contracts](misc%20methods/RESULTS.md), [executed validation](misc%20methods/VALIDATION.md).

**Checks:** recorded isolated-environment infrastructure suite passes eight tests; selected author-code smokes and Mamba data replay pass. Smoke counts are prohibited as paper-performance evidence. NumPy/Brevitas and SciPy compatibility adjustments remain explicit; they do not certify identical production numerics.

**Decision:** retain a research-backed queue. Execute/audit each paper-specific target before transferring it to GKP or claiming its published metrics locally.

### 2026-11-19 — Finite-energy reference, cycle-six event and root-cause plots

**Work:** compare the classical residual approximation with finite-energy oscillator dynamics; investigate cutoff failures, energy growth and rare trajectories. The bounded reference run27ef7fdefecc9020 uses256 trajectories per each of six cardinal states, two cycles, three controllers/settings, cutoffs1536/3072, with dx refinement86ab35fc13213ba0. Some LSTM off-nominal cutoff comparisons remain inconclusive at that budget.

**Follow-ups:** nominal nearest-lattice observations extend to eight cycles and cutoffs3072/4096/5120. The binned 5120 follow-up dd4e949e01da259d completes at2026-11-19T20:52UTC. Its cycle8 mean-fidelity difference versus4096 is -.001281 with95% interval[-.006468,.003905], inside ±.01. This supports mean-fidelity cutoff stability at that tolerance, not energy-tail convergence.

**Finding:** selected trajectory1659 jumps from46.53 photons at cycle5 to3080.67 at cycle6 even at5120. This is not evidence of a universal physical failure immediately after cycle5. Norm and cycle-end means missed important tail/substep behavior. Initial reports left the origin unresolved; the next intervention provides stronger evidence.

**Evidence:** [stage record](docs/qutip_reference_next_stage.md), [root-cause HTML](results/qutip_reference/root_cause_review_20261119/index.html), [plot generator](experiments/plot_qutip_root_cause.py). Historical bin lookup is our numerical representation; QuTiP serves as an independent explicit-circuit control in the next experiment.

### 2026-11-19 — Explicit QuTiP/local single-measurement benchmark

**Question:** is the local conditional update accurate against an explicitly evolved two-mode SUM/homodyne circuit?

**Status/configuration:** completed run5f1d5766c1aa2659,24 circuit configurations and48 final per-input checks; q and p separately, six displaced cardinal inputs, data cutoffs64/96, ancilla256/512/768, sigma_m/L=.05/.16. Two pilots and interrupted attempts remain separate. Primary metric is reference-weighted conditional infidelity; outcome TV is a separate diagnostic.

| Local measurement arm | Worst outcome TV | Worst weighted conditional infidelity |
| --- | ---: | ---: |
| Bins dx=.003 | .002139734 | 2.16787e-5 |
| Bins dx=.0015 | .001024482 | 5.38942e-6 |
| Smooth Mehler resource | 1.20899e-6 | 3.85760e-10 |

**Interpretation:** finer bins improve agreement; smooth evaluation agrees much more closely. Small differences in average energy/logical fidelity do not certify conditional-state agreement. This experiment validates finite truncated data operators, not converged physical memory; boundary mass reaches5.79% in the deliberately small bases.

**Checks/evidence:** recorded16 tests pass; raw hashes/source snapshots and scalar density-curve metrics are checked. [Results](docs/qutip_engine_benchmark_results.md), [protocol](docs/qutip_engine_benchmark.md), [HTML](results/qutip_engine_benchmark/main/5f1d5766c1aa2659/review/index.html).

**Decision:** build the matching continuous sampler, then intervene on long-horizon dynamics with substep logging. Smooth batch timing is12–19× faster than explicit circuit (median16.9×) under that benchmark's exclusions; this is not production decoder latency.

### 2026-11-19 — Smooth-resource and sampler accuracy decision

**Question:** should new reference runs adopt the continuous resource and matching sampler based on independent accuracy and numerical-stability metrics?

**Status:** complete, run06c34206dd1a1b00,569.3s wall time. Frozen protocol and raw/source hashes preserved. Pilots are isolated from the final run; strict familywise DKW and substep gates were fixed before main observations.

**Fresh accuracy design:** seed26111923, six displaced cardinal states × q/p × two resource widths =24 held-out cases; data cutoff96; ancilla512→768. Gate A≥.99999, outcomeTV≤1e-4. These tolerances and reference weighting are project choices.

| Arm | Minimum observed A | Maximum outcome TV | Verdict |
| --- | ---: | ---: | --- |
| Historical bins dx=.003 | .999978529259384 | .00207509615 | Fails accuracy budget |
| Smooth + continuous sampler | .9999999996149406 | 7.64243908e-7 | Passes |

Observed refinement-sensitivity summary gives minimumA=.9999991713695143, still above the budget. This is not a rigorous infinite-cutoff bound. Inverse-CDF residual≤1.9984e-14 against1e-10 tolerance. All24 empirical sampler checks pass the strict familywise95% DKW–Massart bound,65,536 draws/case; maxCDFdistance=.0063717252 versus bound=.0072381312.

**Eight-cycle design:** nominal(.20,.05,.90), dataDelta=.30, nearest-lattice feedback,512 trajectories per each of six cardinal states (3,072/cutoff), seed26111901, cutoffs3072/4096/5120. Frozen binned controls use identical displacement arrays with verified hashes. Measurement histories can differ. Log every displacement, q/p measurement and feedback stage.

| Cutoff | Cycle-end numerical horizon | All logged substeps pass through8? | Smooth F_logical(8) |
| --- | ---: | --- | ---: |
| 3072 | 6 | No | .8761603, diagnostic beyond validity |
| 4096 | 8 | Yes | .8815782 |
| 5120 | 8 | Yes | .8815660 |

Population gates: mean top-decile mass<1e-4 and every trajectory<.01 at every recorded stage; norm tolerance1e-7. Every pointwise4096-to5120 paired fidelity interval lies inside±1 percentage point. At cycle8,5120−4096=-.001219 percentage points,95%interval[-.465922,+.463485].

At5120 smooth−binned F8=+.188692 percentage points,95%interval[-.115726,+.493111]. **No established logical-memory improvement:** interval includes zero. Adoption rests on implementation accuracy and numerical validity.

**Root-cause evidence:** the catastrophic selected event occurs at the p measurement in cycle6. At5120, the binned replay reaches3080.67 photons versus216.82 for smooth. Binned trajectory1659 reaches1.5687% boundary mass after q feedback in cycle7, missed by the old cycle-end checks. Smooth's worst substep boundary mass at5120 is.00151684%. Other hot trajectories and mean energy growth remain; no damping/stabilization was added. The intervention changes representation and sampler jointly, so their separate causal contributions are not identified.

**Implemented decision:** new reference CLI defaults to smooth; explicit --measurement-resource binned preserves legacy mode. Smooth/binned directories and config identities differ; cross-resource comparisons cannot be labeled cutoff refinements. Historical wrappers request binned explicitly. The original local bin implementation and historical raw data remain unchanged.

**Checks/evidence:** recorded23 tests pass in the continuous-ancilla/engine/reference suites; both CLI modes smoke successfully, with deliberately tiny96-level population failures correctly reported. [Decision](docs/smooth_ancilla_results.md), [frozen-design document](docs/smooth_ancilla_experiment.md), [result](results/smooth_ancilla_experiment/main/06c34206dd1a1b00/result.json), [HTML](results/smooth_ancilla_experiment/main/06c34206dd1a1b00/review/index.html), [runner](experiments/smooth_ancilla_experiment.py), [analysis](experiments/analyze_smooth_ancilla.py).

**Rebuild saved-data analysis, without resimulation:**

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.analyze_smooth_ancilla results/smooth_ancilla_experiment/main/06c34206dd1a1b00

**Scope:** fixed-input finite-reference agreement, plus nominal nearest-lattice eight-cycle checks at high cutoffs. Not a50-round quantum decoder grid, Bayes/LSTM quantum superiority, hardware validation or proof that every rare event is numerical.

### 2026-11-19 — Notebook, original notes and persistent logging instruction

**Purpose/status:** completed retrospective consolidation requested by the user for a future paper. Created this notebook, preserved the supplied notes, and added [AGENTS.md](AGENTS.md) requirements for future logging. Read current run artifacts rather than inferring reproduction completion from launches.

**Citation correction:** the earlier response and active smooth-result/report text misattributed PRX Quantum2,020101 to Terhal/Conrad/Vuillot. Its correct authors are **Arne L. Grimsmo and Shruti Puri**, verified against the [publisher](https://doi.org/10.1103/PRXQuantum.2.020101). Correct active result documentation and generated report; leave the frozen preregistration/source snapshot intact and attach this correction. No metric or experiment changes follow from this bibliographic correction.

**Other corrections/limits:** identify the user's Imperial reference as Wan2020; correct lattice peak units; distinguish independent additive scalar readout from finite-energy q/p extraction; retain failed/partial reproductions and the coverage limits. Original notes remain an explicitly unverified historical source where their interpretations have not been checked.

**Validation for this documentation task:** 91 local Markdown evidence links/targets checked across the notebook, instructions, original notes and active smooth-results document; exact counts and selected metrics cross-checked against saved JSON. Regenerated the smooth HTML from saved observations with the corrected citation; analysis reverified frozen source/raw hashes and recomputed saved scalar metrics. Confirmed the frozen protocol retains its original wording and the adoption decision is unchanged. Added discovery links from RESULTS.md and docs/findings.md. No new training or simulation requested or run. Historical test counts above are carried from completed reports, not presented as newly executed tests.

### 2026-11-19 — Remaining simulation-quality audit

**Question/status:** user asked what other quality issues remain after the smooth-resource decision. Completed a read-only review of current simulation, scoring, gate and analysis code plus saved results. This is a scoped audit, not an exhaustive code review or a new physical benchmark. No additional numerical defect is established by the observations below.

**Newly identified workflow gaps:**

1. **Substep gates have not reached the ordinary reference runner.** The dedicated smooth experiment records energy/boundary mass before displacement, after displacement, and after each measurement/feedback. The ordinary [reference simulation](experiments/qutip_reference.py) observes norm and boundary population at cycle ends and builds its numerical gate from those observations. Selecting smooth mode does not automatically inherit the dedicated experiment's stronger stage checks. The prior binned transient boundary violation demonstrates why this matters; it does not prove the new smooth default has an unobserved violation. Propagate stage logging/gates before extending other controllers/noise settings. Even the dedicated logger combines q/p displacement into one recorded stage, so these are checks of recorded stages rather than a continuous-time guarantee during gates.
2. **Logical-score refinement is not closed for the new long smooth run.** The run fixes192 logical quadrature bins. It saves32 final states per cardinal input (192 columns at each cutoff), but [its analyzer](experiments/analyze_smooth_ancilla.py) does not rescore those snapshots at384 or768 bins. The older two-cycle binned reference has a separate192→384 check; that cannot certify these newly evolved hot states. The extraction-trace check constrains diagonal completeness, not convergence of every logical coherence integral. This is an unclosed accuracy check, not evidence that the reported88.1566% score is wrong. Use the saved snapshots, including targeted hot-state coverage, to refine logical scoring before a stronger accuracy claim; final snapshots alone cannot check intermediate-cycle scores.

**Remaining physical/model limits:**

- The completed smooth reference uses ideal SUM interactions, ideal homodyne, fresh resources, instantaneous feedback and no loss/dephasing/stabilization. [Channel implementations](quantum/channels.py) and independent tests exist, but the completed reference does not call them. Mean photon number at5120 rises from5.070557 initially to260.790151 atcycle8; final99th percentile≈1053.013 and maximum≈2277.337. Removing the discretization spike did not create a stabilized memory. Interpret this as the chosen unstabilized protocol, not a hardware lifetime. Realistic oscillator loss/dephasing/control imperfections are discussed in [Grimsmo and Puri2021, Sec.II C, Eq.(16)](https://doi.org/10.1103/PRXQuantum.2.020101). Adding loss later must avoid double-counting displacement noise already calibrated from loss and must not be used merely to hide numerical heating.
- Classical additive Gaussian readout and finite-energy ancillary extraction are distinct channels. In the latter, changing sigma_m also changes the Mehler resource's envelope/energy and measurement back-action. Matching isolated peak width does not match the full conditional instrument. Consequently decoder rankings across engines can reflect channel mismatch. The Gaussian Bayes controller in the quantum reference is not an optimal quantum filter.
- The explicit QuTiP accuracy control covers one q OR p extraction with fixed input states and data operators at64/96 levels (fresh confirmation uses96). Its implementation-independent circuit/projection is valuable, but input preparation, feedback and logical scoring are shared. It does not independently verify a full5120-level eight-cycle simulation or its high-energy conditional states. Reference-weighted fidelity is an average over outcomes and the reported minimum is over tested input cases; neither is a worst-case channel bound.
- Large-basis mean fidelity and stage boundary gates pass through eight nominal nearest-lattice cycles at4096/5120. This does not certify3072 aftercycle6, later cycles, every trajectory's energy convergence, other noise points, or LSTM/Bayes quantum rankings. Registered±1pp cutoff equivalence and sampling intervals do not support arbitrary extra decimal places. The primary conditional-state fidelity near100% and logical memory fidelity≈88.16% are different metrics.
- Noise coverage remains synthetic and limited: 61 stationary plane intersections, not the full125-point cube; the long smooth validation uses one parameter setting/controller. The stationary/quantum noise assumptions exclude cross-quadrature correlations, negative-rho tests in this grid, arbitrary heavy-tailed drift and hardware-calibrated measurement imperfections. Classical nonstationary infrastructure is separate from completed quantum validation.
- Rare-event resolution is limited. With zero occurrences of a specified event in3072 independent eight-cycle trajectories, the one-sided exact95% binomial upper probability is 1−.05^(1/3072)=.00097469794 per eight-cycle trajectory (about.0975%), not zero. This calculation assumes a fixed event and the tested equally weighted input ensemble; the historical1000-photon jump threshold was diagnostic, not preregistered for adoption. The heatmap's1024 trials/cell cannot establish1e-5-level failure rates. More initialization seeds do not replace independent noise samples. Existing memory intervals are pointwise and exclude model uncertainty.
- Six-cardinal average logical fidelity is not worst-case fidelity. The modular gauge-trace convention is a legitimate stated choice, but different recovery/readout conventions need not produce the same score; [Pantaleoni, Baragiola and Menicucci2020, Eqs.(3)–(4)](https://doi.org/10.1103/PhysRevLett.125.040501) define its meaning. Published-target conventions remain unresolved for Wan and the Convy control, and Baireuther's numerical target remains unmatched; these cannot serve as completed external validation of the whole simulator.

**Evidence/validation performed:** inspected [local Fock/scoring code](quantum/qutip_reference.py), [ordinary runner](experiments/qutip_reference.py), [instrumented runner](experiments/smooth_ancilla_experiment.py), [explicit circuit control](quantum/measurement_benchmark.py), [smooth analyzer](experiments/analyze_smooth_ancilla.py), and existing tests/protocols. Read saved resultJSON and NPZ snapshot shapes at3072/4096/5120; recomputed the zero-event bound analytically. No training, new trajectories, high-cutoff scoring calculation or test suite was run for this audit. Physical references were checked against primary sources; code/model limitations above are local observations and their stated implications.

**Verdict/decision:** retain the existing smooth adoption within its registered scope. Prioritize (1) carrying stage gates into the ordinary runner and (2) logical-score refinement on saved smooth states, then extend independent high-energy validation. Add loss/dephasing and stabilization only as separately defined physical interventions; broaden noise/controller coverage and rare-event budgets before hardware or general robustness claims.

### 2026-11-19 — Simulation-quality infrastructure implementation

**Question/status:** Implemented and verified within the scope below. User requested complete infrastructure for the quality-audit items before benchmarking. Scope: stage gates in normal reference execution, memory-bounded logical rescoring and representative/hot-state snapshots, explicit physical-channel/detector/control settings, independent supplied-state measurement validation, expanded noise coverage, and resumable benchmark plans with fixed-budget rare-event statistics. Production benchmarks remain Planned; bounded software checks are authorized as implementation verification.

**Design:** preserve historical raw runs and compatibility entry points. Record model/intervention distinctions, source/config hashes and separate numerical versus physical/energy/event statuses. Finite-duration decoherence in the scalable runner will be labeled a split/lumped approximation and checked separately from the independent explicit circuit; passive cooling is not advertised as a GKP stabilizer. No hardware-calibrated performance or rare-event resolution is inferred from infrastructure tests.

**Implementation outcome:** the standard CLI now uses an instrumented schema2 engine; the old simulation() entry point remains a compatibility control. Each primitive displacement, noise, measurement, delay, feedback and rotation is logged and gated. It saves representative and selected hot states every cycle. Added streaming192/384/768 scoring, a supplied-state independent instrument validator, an effective-noise calibration contract with disjoint split IDs, full-cube/negative-memory/correlated/Student-t/scheduled noise options, optional physical loss/dephasing/detector/control settings, and explicit coupled two-mode QuTiP finite-duration gates alongside the labeled scalable approximation.

**Benchmark infrastructure:** frozen source/config/checkpoint hashes, atomic sealed raw arrays, duplicate-writer locks, resumable fixed-budget chunks, paired cutoff intervals, state-stratified Clopper–Pearson/Bonferroni event bounds, separate scoring/instrument coverage, storage preflight and HTML/PNG/SVG/CSV/JSON reports. Parameter matrices support project-selected interventions without mutating completed runs. Six draft benchmark plans are saved and remain unrun; their exact counts/configuration/estimated storage are in the verification record.

**Checks/results:**50 final targeted tests passed in8.13s, including22 dedicated infrastructure cases plus resource/reference/channel regressions. The new tests cover negative controls as well as nominal behavior. CLI smoke completed2/2 jobs by partial execution and resume; both pass saved-state scoring refinements but fail population gates at deliberately small32/48-level bases, correctly reported as diagnostic. A separate6-level ideal-instrument control passes and cannot certify those larger bases. The normal24-level CLI now exposes stage failures and its saved-state refinement passes. Synthetic calibration CLI also completes. The additional historical nonstationary regression was stopped after439.15s with58 passing checks and no reported failures; its unchanged adaptive-filter stress path had not completed, so the broader suite remains incomplete and those counts are not added to the50. No production trajectories or high-cutoff rescoring were launched.

**Evidence/next step:** [runbook](docs/simulation_quality.md), [verification](docs/simulation_quality_validation.md), [smoke HTML](results/simulation_quality_infrastructure/v1/smoke/index.html). The earlier audit's missing normal-runner stage gates and missing rescoring capability are addressed as infrastructure; the old high-cutoff smooth run still needs its actual refinement execution. The remaining physics, high-energy and generality claims require the planned benchmarks. Passive cooling remains distinct from a paper-specific GKP stabilization protocol.

### 2026-11-19 — Local HTML hosting

**Purpose/status:** user requested local HTML access. Started the standard-library HTTP server bound to127.0.0.1:8765, serving qkp/; no public deployment or new benchmark. The infrastructure smoke report is at http://127.0.0.1:8765/results/simulation_quality_infrastructure/v1/smoke/ .

**Command:** .venv/bin/python -u -m http.server 8765 --bind 127.0.0.1 --directory /Users/christopherbabayan/programs/QUANTUM/qkp . The active process is retained in the local execution session. Verified HTTP200 for HTML, PNG/SVG figures, eventCSV and summaryJSON; report title/content match. Existing smoke/benchmark status is unchanged. An attempted browser-opening action timed out; the HTTP endpoint is verified independently.

### 2026-11-19 — Clarification of the locally hosted smoke plots

**Question/result:** user asked why the hosted graphs contain only three data points. Checked the saved smoke plan and summary: the infrastructure smoke runs two correction cycles, so fidelity/energy curves contain cycle0,1,2. Each cutoff has12 trajectories (two per cardinal state), and32/48-level bases fail population gates as expected for these bounded software checks. These are not the completed eight-cycle scientific benchmark curves. The separate completed smooth-accuracy study has eight cycles plus initialization (nine cycle samples),512 trajectories/cardinal state and3072/4096/5120 cutoffs; its report remains available. No additional simulations were launched in answering this question. The rare-event panel has one aggregate estimate/bound per configuration, not one point per cycle.

### 2026-11-19 — Authorized production launch and EC2 execution

**Question/status:** user authorized production benchmarks, then selected EC2 with metric fetching and AWS AdministratorAccess account882574060060. Initial local storage check found approximately2.8GB free versus278.6GB estimated uncompressed outputs across the six suites. No large local production job was launched; EC2 preparation proceeds in the selected account/us-east-1.

**Independent pre-scale check:** the experiment-plan workflow requires an independent documented control/intervention pilot. The validator completed controladaa00c49f057046 and combined-intervention1d32cca10210d308, each12trajectories/2cycles/N32/seed26111973. Both verify current sources,25stages, raw seals, matched exogenous draws and decoder-only access to reported syndromes. Both fail population gates from initialization, as expected at this deliberately small cutoff; norm errors<2.1e-14. No software blocker was found. [Independent audit](results/simulation_quality_infrastructure/independent_pilot_20261119/audit.json), [commands](results/simulation_quality_infrastructure/independent_pilot_20261119/commands.txt). This is software validation, not physical benchmark evidence.

**Remote design:** one new isolated m7i.2xlarge (8vCPU/32GiB),400GiB encrypted gp3, private S3 artifacts, no inbound ports and SSM management. Pinned Python3.13.13 and scientific versions have Linux wheels; CPU Torch is used. AWS Pricing API quoted computeUSD.4032/hour, effective2026-11-01. Initial compute safety cutoff24hours, with early stop on completion/error; retained EBS/S3 remain storage resources. Source/checkpoint bundle hashes, actual remote environment, baseline checks, trajectories, scoring and incomplete numerical/instrument checks are recorded. Metrics/figures are fetched separately from large raw arrays.

**Sequence/decision fixed before new observations:** execute the numerical baseline and bounded explicit finite-gate comparison first, then saved-state192/384/768 refinement. Broader physics/noise/coverage/rare suites proceed only if the numerical baseline's registered stage, scoring and paired-cutoff checks pass; a failure/inconclusive baseline keeps those sweeps queued. Preserve failed observations and numerical labels. The remote platform/env differs from macOS and is recorded as a new run, not silently merged into historical data.

**Launch and observed setup:** instancei-0be3c3be10368ba15 is running and SSMOnline. Runqkp-20261120-003030-d2292a uses a5,294,028byte source/checkpoint archive. Remote Python3.13.13/NumPy2.5.5/SciPy1.18.1/QuTiP5.3.1/Torch2.15.0+cpu installed successfully. The50 targeted tests passed on Linux in16.86s before scaling. At the first fetched active status the numerical stage was running and finite-gate controls were7/8jobs completed; those are progress observations, not final benchmark conclusions. Local production remains unlaunched.

**Monitoring/evidence:** private S3 retains raw and small metric trees separately. A read-only live exporter adds per-decoder curves without modifying scientific execution; its separately deployed digest is recorded alongside the original source bundle in [manifest.json](results/ec2_runs/production_20261119/manifest.json). A local30s fetch loop is active. [Remote validation](results/ec2_runs/production_20261119/metrics/remote_validation.json), [environment](results/ec2_runs/production_20261119/metrics/environment.json), [operations/run details](docs/ec2_production.md). Local live report: http://127.0.0.1:8765/results/ec2_runs/production_20261119/metrics/live/ . Numerical/scoring/coverage results remain pending until their artifacts complete.

**First completed stage:** the fetched finite_gate summary reports8/8jobs executed and8stage-gate failures at its deliberately small32/48-level bases. These are retained bounded finite-operator diagnostics, not validated memory-performance results. The large-basis numerical baseline is still running; broader sweeps remain queued. HTTP200 was verified for the production suite page, live page, statusJSON and remote validation receipt. No claim of completed numerical/physics/coverage/rare production results is made at this point.

### 2026-11-20 — EC2 production progress and completed numerical baseline

**Question/status:** User requested a fresh status check. At 19:39 UTC (11:39 PST), instance `i-0be3c3be10368ba15` was running with SSM Online. Run `qkp-20261120-003030-d2292a` has passed the numerical prerequisite and entered production sweeps. This entry supersedes the preceding launch-time baseline-running status; the full benchmark remains incomplete.

**Configuration/method:** Same frozen source, checkpoint, environments, seeds and registered criteria as the [EC2 manifest](results/ec2_runs/production_20261119/manifest.json) and [run record](docs/ec2_production.md); no simulation parameters or acceptance thresholds changed. Numerical baseline: nominal Gaussian displacement/readout scales sigma_q/L=.20, sigma_m/L=.05, correlation rho=.90; smooth resource, Delta=.30; eight cycles; 512 trajectories per each of six cardinal states (3,072 per decoder/cutoff), cutoffs 4,096/5,120, nearest-lattice, Gaussian Bayesian and frozen LSTM256×3 seed0. Existing modular gauge-trace logical-fidelity convention and cited methods remain as registered in [simulation quality methods](docs/simulation_quality.md).

**Completed numerical result:** 2/2 jobs, zero stage-gate failures; both saved-state 192/384/768 scoring refinements pass, and all three paired cutoff-equivalence checks pass at every cycle (95% interval wholly within ±.01). At cutoff5,120, cycle8 six-cardinal mean logical fidelity and pointwise95% interval are nearest .890996 [.880293,.901699], Bayesian .988344 [.985364,.991324], LSTM .985122 [.981550,.988693]. These are logical-memory metrics, not classification accuracy or hardware fidelity. Supplied-state independent instrument coverage is still absent for these production states, so this is a passed registered numerical sample check, not universal numerical certification or completed external validation.

**Partial results/uncertainty:** Latest suite summaries contain physics13/20 and noise13/14 executed jobs; the thirteenth job in each still needs refinement, while the root heartbeat reports12 fully processed jobs each. All observed physics/noise primitive stage gates pass. Completed physics comparisons for nearest-lattice with loss, and loss+dephasing, do not establish the registered cutoff equivalence: intervals marginally cross ±.01 (loss cycles5,7,8; combined cycle6). Point estimates remain inside the margin; interpret these as inconclusive equivalence, not demonstrated large basis bias. All18 completed noise cutoff comparisons pass. Finite-gate diagnostics remain8/8 executed with8 population-gate failures at deliberately small32/48-level bases. Coverage0/250 and rare-event0/300 remain queued; do not infer their robustness or rare-event bounds from current observations.

**Evidence/checks actually executed:** Queried EC2/SSM state, synchronized small S3 metrics, read [numerical summary](results/ec2_runs/production_20261119/metrics/numerical/summary.json), [physics summary](results/ec2_runs/production_20261119/metrics/physics/summary.json), [noise summary](results/ec2_runs/production_20261119/metrics/noise/summary.json), and [status heartbeat](results/ec2_runs/production_20261119/metrics/status.json). Local live HTML returned HTTP200 and contains92 completed decoder rows across suites. No fresh trajectories, tests, rescoring, or changes to the running worker were launched during this check.

**Decision/next step:** Leave the existing worker running and retain all invalid/inconclusive controls. The configured24-hour shutdown is around00:32 UTC on21November (16:32 PST on20November); the remaining full coverage/rare grids cannot finish within this original window at observed throughput. Preserve partial results and raw S3 checkpoints; continuation and resolution of inconclusive comparisons require subsequent execution, not a claim that the production suite is complete.

### 2026-11-20 — GitHub repository refresh

**Purpose/status:** User authorized updating GitHub while EC2 continues. Prepared the accumulated source, tests, reproduction workspaces, HTML dashboards, compact evidence, and research documentation for the existing `chen-tin/qkp` origin on main. The remote was fetched before preparation; no divergent commits were present. Commit and remote history provide the synchronization receipt.

**Scope/provenance:** Added a repository README with setup, methods, report navigation, data availability and current limitations. Added ignore rules for multi-gigabyte raw NPY/NPZ datasets, training/resume weights under results, logs, live EC2 mirrors, local launch payloads and duplicate ZIP exports. Preserved these local files; none were deleted. Included the frozen5.3MB LSTM256×3 seed0 production controller required by the EC2 bundle. Kept compact historical manifests/configurations/summaries, source snapshots and figures in Git. Original immutable artifacts were not edited.

**Production evidence:** Captured [a dated read-only snapshot](results/ec2_runs/production_20261119/snapshots/20261120T194213Z/README.md) at19:42:13UTC, with per-file checksums in snapshot_manifest.json. The snapshot is explicitly partial, with independently exported adjacent heartbeat/report timestamps; it does not turn GitHub into a live AWS feed. Registered numerical sample checks passed, finite-gate controls remain invalid, the two nearest-lattice physics cutoff comparisons remain inconclusive, and full production coverage/rare-event work remains incomplete. Raw production observations remain in the existing private S3 archive and are not claimed as publicly available.

**Checks actually executed:** Fresh local verification of the ten newly added test modules covering Baireuther, Convy, Wan, LSTM sweep recovery, channels, smooth extraction, explicit instrument, finite-energy reference and simulation quality: **71passed in9.61s**, using one BLAS/OpenMP thread. This is bounded software validation, not a rerun of completed scientific benchmarks or a production completion claim. No new EC2 resources, trajectories or full training runs were started. Verified56 snapshot file checksums, HTTP200 with full response bodies for both offline dashboards and captured production plots, and no staged file above50MB. Credential-pattern review found only19 false matches within verified base64 image data. Added binary/generated/vendor Git attributes to preserve frozen evidence bytes and make source diffs readable; the final staged whitespace check passes. Normalized the formerly empty readME.md entry to README.md for GitHub and removed one extra blank line from an active paper-workspace requirements file. Git transfer is checked separately before publication.

**Decision:** Publish source, documented limits, offline HTML and compact evidence through the user-authorized GitHub update; retain large raw data outside Git. Further benchmark findings require a later dated snapshot and notebook update.

**Publication receipt:** At19:45:59UTC, `git ls-remote origin refs/heads/main` confirmed published commit `7abb58a25046056c06038a6178171c3ca666791b` on main, and the working tree matched origin. The first push using the default work-account credential received403; the already signed-in personal account `babayChris` has push permission and completed the push through a command-scoped credential helper. The default signed-in account and global Git credential configuration were not changed. This follow-up records the verified publication; it does not change any benchmark observations.

## Paper preparation: remaining decisions and evidence gaps

1. Define the main contribution: decoder robustness, simulator validation, or a staged study combining them. No novelty or state-of-the-art claim has been established by this notebook.
2. Audit older E9–E15 logs, checkpoints and raw records into a consistent per-study completion table. Do not reuse stale “built/not run/running” status as current evidence.
3. Resolve Convy control/scoring differences and Wan scoring/channel conventions; investigate Baireuther's numerical mismatch using new diagnostic protocols without retuning on the finished test set.
4. Hold architecture fixed while varying training-noise coverage; reserve new noise trajectories/settings for confirmatory evaluation. Decide whether to fill the125-point cube and add negative correlations or physical noise beyond stationary Gaussian displacements.
5. Extend validated smooth dynamics to other controllers/noise settings and longer horizons only with fresh numerical checks. Separate loss/dephasing, stabilization and hardware calibration as physical interventions.
6. Select additional architectures based on runnable, audited paper-specific evidence and compute budget. Keep original-task reproduction separate from transfer.
7. Pin citations/versions, equation locators, fit conventions, split seeds, software, raw hashes and figure generation for every manuscript claim. Audit existing manuscript text and generated sites against current conclusions.
8. Preserve local-only/ignored results in a durable reproducibility package before submission. Decide the repository/archive and licensing details when publication is authorized.

## Template for future work entries

### YYYY-MM-DD — Descriptive work item

**Question / purpose:**

**Status:** Planned / Running / Interrupted / Completed / Inconclusive; specify whether infrastructure, diagnostic, author-data reanalysis, candidate reproduction, or audited reproduction.

**Method and sources:** primary citation, version, equation/figure/table or author-code locator; distinguish published method from project choices.

**Configuration and data:** code/noise model; exact distribution/units/correlation/schedule; all varied/fixed parameters; rounds/trials/seeds; train/validation/test identity; checkpoints; source/config/environment manifest and hashes; run path/command.

**Metric and decision rule:** formula/convention, denominator, horizon, aggregation, direction, uncertainty/multiplicity, numerical gates, and acceptance criterion fixed before observations. Label exploratory choices.

**Result and evidence:** measured values, raw counts, intervals/sensitivity, null/negative outcomes, failures and exclusions; link raw results, plots, report and analysis code.

**Checks actually executed:** command, outcome, coverage and limitations. Separate software faults from valid model failures.

**Interpretation and limits:** what changed in our belief, alternatives still unresolved, and the exact demonstrated scope.

**Decision / next step:** implemented action or remaining question; identify any earlier entry superseded by this one.
