# E13: learned-decoder study (infrastructure; not yet run)

**Question.** At the nominal point the LSTM's logical error is 3–4× that of matched Bayes,
although matched LSTMs come within 2–17% of Bayes where errors are common. Is this gap due
to model size, the amount of training data, the rarity of informative (error-carrying)
sequences, or the LSTM architecture? E13 also measures the seed spread of the single-seed
models (LSTM-matched, LSTM-switching).

## Pieces

| File | Contents |
|---|---|
| `decoders/learned.py` | `ParityGRU`, causal `ParityTransformer` (sinusoidal positions; closed-loop state is the feature prefix), `SeqDecoder` (closed-loop reference), `FastSeq` (whole-sequence fast path), `load_model` (reads `arch`; old checkpoints load as LSTM), `matched_kwargs` (parameter matching) |
| `training/train_variants.py` | variant registry, cost estimate (`--cost`), and training driver (reuses `train_lstm` evaluation and head selection) |
| `experiments/run_learned.py` | E13 evaluation via `run_all.evaluate_point`, and `--analyze` (table: P_L(50), time-averaged error, ratio to matched Bayes, parameters, training seconds) |
| `tests/test_learned.py` | shapes; incremental state = full sequence; causality; closed loop = open loop = fast path for GRU and transformer; parameter matching; old-checkpoint loading; tiny training run |

## Variant grid (19 variants and 6 extra seeds)

| Group | Variants | Isolates |
|---|---|---|
| size | LSTM hidden 32, 64, 128, 256 (200k sequences) | model capacity |
| data | LSTM-128 with 25k, 50k, 100k, 200k, 400k sequences | training-set size |
| curriculum | LSTM-128 `rare`, seeds 0–2, against `none`, seeds 0–2 | rarity of informative sequences |
| arch | GRU (hidden 148) and transformer (hidden 92, 2 layers, 4 heads), about 201k–207k parameters, matched to the LSTM-128's 201k; seeds 0–2 | architecture |
| seeds | LSTM-matched (ρ = 0.5, σ_q = 0.26L) and LSTM-switching, seeds 1–2, trained by `train_lstm.train_one` under their usual names | seed spread |

**Curriculum `rare`:**
- **Noise annealing.** A second pool is drawn at 1.3× σ_q, where nearest-lattice jumps are
  common. Its share of each epoch falls linearly from 0.5 to 0 over the first half of
  training; the second half is nominal only.
- **Oversampling.** Nominal sequences that contain at least one odd jump are drawn with
  weight 4.
- **Validation** always uses plain nominal data, so head selection and early stopping
  measure the target task.

## Evaluation points

All are 50 rounds, against matched Bayes.

| Point | Noise | Why it is included |
|---|---|---|
| nominal | σ_q = 0.20L, σ_m = 0.05L, ρ = 0.9 | the rare-error gap |
| rho0.5 | ρ = 0.5 (E2 level A) | where the matched LSTM is within 2% |
| sq0.26 | σ_q = 0.26L (E3) | where the matched LSTM is within 17% |
| switching | E6 regime switching | against regime-aware Bayes |

Every variant runs at every point. Test sets grow up to 60 000 trials.

## Cost estimate

Not run. Scaled from the measured 339 s for LSTM-128 on 200k sequences: linear in the number
of sequences and in the parameter count, with an architecture factor. `python -m
training.train_variants --cost` prints the table. Total **about 2.5 h** of training on the
laptop RTX 5080 at full power. The largest items are LSTM-256 (about 22 min), LSTM on 400k
sequences (about 11 min) and the three transformers (about 9 min each). Evaluation is cheap,
because LSTM inference is about 0.2 µs per trial-round on the fast path.

## Running

```bash
python -m training.train_variants            # train the variants and the extra seeds
python -m experiments.run_learned            # evaluate (LSTMs are deferred until trained)
python -m experiments.run_learned --analyze  # results/E13_summary.csv
```
