# Trained checkpoints

Key checkpoints of the two published experiments, so the probing analyses can be
reproduced **without retraining** (~4 h on an A100 each). Both were trained with
`scripts/train_bounce2d.py` defaults except where noted; the *only* variable
manipulated between the two is `sim_coeff_t`.

| Run | `sim_coeff_t` | Epochs | Checkpoint here | Notes |
|---|---|---|---|---|
| `main_simt12_seed1` (Exp A) | **12** | 300 | `e-300.pth.tar` (epoch 300) | The `latest.pth.tar` of this run was corrupted (interrupted write) — all published analyses use `e-300.pth.tar`. |
| `ablation_simt0_seed1` (Exp B) | **0** | 100 | `latest.pth.tar` (epoch 99) | Final checkpoint of the 100-epoch ablation run. |

Common settings: `n_base=4000`, `batch_size=256`, `cov_coeff=12`, `std_coeff=16`,
`std_margin=2.0`, `idm_coeff=0`, `speed_range=(0.02, 0.10)`, `T=16`, `img_size=65`,
`seed=1`, bfloat16 + AMP.

## Provenance caveats

* **`main_simt12_seed1/config.yaml` is stale.** After the 300-epoch Exp A run
  finished, an aborted restart was launched in the same folder with ablation
  settings and overwrote `config.yaml` (it reads `sim_coeff_t: 0, epochs: 100`).
  The authoritative Exp A settings are the ones above (`sim_coeff_t=12`,
  300 epochs); `e-300.pth.tar` itself predates the restart and is from the real
  Exp A run. `ablation_simt0_seed1/config.yaml` is accurate.
* The full checkpoint sweeps (every 10 epochs) are too heavy for git and live on
  the original Google Drive run folders. The Exp A Drive folder additionally
  contains stale files from other runs sharing the folder name (`e-0/10/20/30`
  from the aborted restart, `e-400` from an older run) — for Exp A, only
  `e-100`, `e-200`, `e-300` belong to the published 300-epoch run, which is why
  `scripts/run_probes.py` has a `--max-epoch` flag.

## Loading a checkpoint

```python
import probes as P
model = P.load_checkpoint_model("runs/main_simt12_seed1/e-300.pth.tar", device="cuda")
```

Checkpoints store `model_state_dict`, optimizer/scheduler state, the XY-probe
head, `epoch` and `step` — `probes.load_checkpoint_model` rebuilds the exact
AC-Video-JEPA architecture and loads `model_state_dict`.
