# bounce2D × EB-JEPA — Probing conserved physics in JEPA latents

> Does a JEPA world model encode a **conserved invariant** (kinetic energy,
> `E = ½m‖v‖²`) in its latent — not just instantaneous state like position and
> velocity — and is that encoding causally tied to how the model was trained?

This repository is a fork of
[facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa)
(Energy-Based Joint-Embedding Predictive Architectures — the original library
README is preserved at [docs/EB_JEPA_README.md](docs/EB_JEPA_README.md)),
extended with:

* **`bounce2d/`** — a fully controlled synthetic environment: one ball in a
  box, bit-exact elastic bounces, exact ground truth for position / velocity /
  energy at every frame, plus a paired **physics-violation** eval set
  (teleport, phantom bounce, energy gain, energy loss).
* **An AC-Video-JEPA integration** (`bounce2d_ebjepa.py`) that plugs bounce2D
  into the stock training loop of `examples/ac_video_jepa/main.py`.
* **A probing suite** (`probes.py`) — linear vs quadratic decodability of
  energy, invariant stability over time, and "surprise" (one-step latent
  prediction error) by violation type.
* **Trained checkpoints** for the two published runs ([runs/](runs/README.md)),
  so all analyses can be reproduced without retraining.
* **Full findings** in [docs/bounce2d_results.md](docs/bounce2d_results.md).

## Headline results (Exp A: AC-Video-JEPA, sim_coeff_t=12, epoch 300)

| Probe | R² / ratio | Reading |
|---|---|---|
| Position from 1 frame (linear) | 0.999 | Position is trivially in the latent |
| Velocity from 1 frame (linear) | ≈ 0 | Sanity floor: a single frame has no motion |
| Velocity from frame pairs (linear) | 0.900 | Velocity is linearly decodable from 2 frames |
| Energy from pairs (**linear**) | −0.056 | Energy is *not* a linear read-out |
| Energy from pairs (**quadratic**) | **0.750** | Energy *is* a quadratic read-out — consistent with `E ∝ ‖v‖²` |
| Surprise: teleport / phantom bounce | ×7.46 / ×2.65 | Continuity violations strongly detected |
| Surprise: energy gain / energy loss | ×1.54 / ×0.81 | Conservation violations detected **asymmetrically** |

A trajectory-level triviality test (independent protocol) confirms the energy
signal is non-trivial: direct quadratic probe R² = 0.570 vs −0.044 for a
linear-speed reconstruction, with a permutation control at −0.35
([docs/probing_energy_triviality.md](docs/probing_energy_triviality.md)).

**Interpretation (short version):** the latent carries the velocity vector and
energy is a *quadratic read-out* of it — a necessary but not sufficient
condition for a learned "concept" of energy conservation. See
[docs/bounce2d_results.md](docs/bounce2d_results.md) for the careful version.

## Installation

```bash
git clone https://github.com/Mantra20/eb_jepa.git
cd eb_jepa
pip install -e .            # or: pip install torch numpy scikit-learn matplotlib einops fire omegaconf wandb
```

Verify the integration end-to-end on CPU (~1 min, no GPU needed):

```bash
python smoke_test.py
```

## Reproducing the experiments

### 1. Generate the evaluation set

```bash
python make_data.py         # writes bounce2d_eval.npz (paired normal + 4 violations, seed 777)
```

Training data is generated on the fly by the loader (normal trajectories only —
the world model never sees a violation during training).

### 2. Train (~4 h on an A100 for Exp A)

```bash
# Exp A (main): temporal-similarity regularization ON
python scripts/train_bounce2d.py --exp-name main_simt12 --sim-coeff 12 --epochs 300

# Exp B (ablation): the ONLY change is sim_coeff_t = 0
python scripts/train_bounce2d.py --exp-name ablation_simt0 --sim-coeff 0 --epochs 100
```

Checkpoints are written to `runs/<exp>_seed1/` every 10 epochs. Watch the
per-epoch `eff_rank` (healthy > 5) and `temp_var` (healthy > 0.1) logs — they
are the anti-collapse monitors.

**Skip this step if you just want the analyses:** the two key checkpoints are
already in [runs/](runs/README.md).

### 3. Probe a checkpoint

```bash
# Exp A (its latest.pth.tar is corrupted -> use e-300; --max-epoch guards
# against stale checkpoints if you probe the full Drive folder)
python scripts/run_probes.py --run-dir runs/main_simt12_seed1 --checkpoint e-300.pth.tar --sweep --max-epoch 300

# Exp B
python scripts/run_probes.py --run-dir runs/ablation_simt0_seed1 --checkpoint latest.pth.tar --sweep
```

Prints all four probes and saves `probe_figures.png` (linear-vs-quadratic bar,
surprise curves aligned at the violation step, checkpoint sweep).

### 4. Triviality test (is E encoded beyond v?)

```bash
python scripts/triviality_test.py --run-dir runs/main_simt12_seed1 --checkpoint e-300.pth.tar
```

### Colab

[notebooks/bounce2d_colab.ipynb](notebooks/bounce2d_colab.ipynb) runs the whole
pipeline (setup → config → train → probes → triviality test) on a Colab GPU,
with checkpoints streamed to Google Drive.

## Repository layout (project additions)

```
bounce2d/               the environment
  env.py                physics: elastic box, energy conserved bit-exactly
  render.py             65×65 2-channel frames (anti-aliased ball + wall mask)
  violations.py         paired trajectory generation with 4 violation types
  dataset.py            numpy bundles (train = normal-only, eval = paired)
  sanity.py             loud generation-time correctness checks
bounce2d_ebjepa.py      adapter: bounce2D -> AC-Video-JEPA data pipeline
make_data.py            builds bounce2d_eval.npz (+ sanity checks)
probes.py               the probing suite (see its docstring for the design rationale)
smoke_test.py           CPU end-to-end integration test
scripts/
  train_bounce2d.py     reproducible training launcher (Exp A / Exp B)
  run_probes.py         probes + checkpoint sweep + figures
  triviality_test.py    trajectory-level "E beyond v" test
notebooks/
  bounce2d_colab.ipynb  Colab version of the full pipeline
runs/                   trained checkpoints + provenance notes
docs/
  bounce2d_results.md            full findings & interpretation
  probing_energy_triviality.md   secondary test protocol, bugs & fixes
  EB_JEPA_README.md              original upstream library README
```

Upstream library code (`eb_jepa/`, `examples/`) is unchanged except for two
small additions: a `bounce2d` dispatch in `eb_jepa/datasets/utils.py`, and
per-epoch `eff_rank` / `temp_var` / regularizer-component logging in
`examples/ac_video_jepa/main.py`.

## Design choices worth knowing before you extend this

* **Energy is only probeable across trajectories.** It is constant within one
  (that's the point of conservation), so initial conditions are randomized and
  every probe splits train/test **by trajectory**.
* **The violation suite is orthogonal by construction.** Continuity violations
  (teleport, phantom bounce) change position but preserve energy; conservation
  violations (energy gain/loss) change energy but keep position continuous.
  Surprise on the latter, above the continuity baseline, is the evidence that
  the model tracks the invariant and not just position.
* **The rendering leaks nothing.** Frames contain only the ball disc and the
  wall mask — no velocity or energy is written into the input, so any energy
  signal in the latent was inferred by the model.
* **Linear vs quadratic probes are the physics test.** `E ∝ ‖v‖²`, so if the
  latent encodes the velocity *vector*, a quadratic probe must beat a linear
  one on energy. That gap (−0.056 vs 0.750) is the core finding.

## Open directions

* Cross-architecture comparison (Image-JEPA / Video-JEPA / AC-Video-JEPA) on
  the same pipeline — planned, not yet run.
* Proper A/B causal ablation of the IDM term (`idm_coeff`), analogous to the
  `sim_coeff_t` ablation.
* A unified protocol reconciling the per-frame (R² = 0.750) and
  trajectory-level (R² = 0.570) energy measurements into a single number.

## Attribution

Built on [EB-JEPA](https://github.com/facebookresearch/eb_jepa) by Meta AI
Research (FAIR) — see [docs/EB_JEPA_README.md](docs/EB_JEPA_README.md) and
[LICENSE.md](LICENSE.md). The bounce2D environment, probes, and analyses were
developed for the "Hack the World(s)" hackathon (June 2026) and after.
