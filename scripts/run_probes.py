"""Run the full probing suite on a trained bounce2D checkpoint (+ optional sweep).

Reproduces the analysis of the published runs:

    # Exp A (latest.pth.tar of that run is corrupted -> use e-300)
    python scripts/run_probes.py --run-dir runs/main_simt12_seed1 --checkpoint e-300.pth.tar --sweep

    # Exp B
    python scripts/run_probes.py --run-dir runs/ablation_simt0_seed1 --checkpoint latest.pth.tar --sweep

Probes (see probes.py for the full rationale):
  1. instantaneous  -- position from a single frame (expected R2 ~ 1),
                       velocity from a single frame (expected R2 ~ 0, the baseline)
  2. temporal       -- velocity (linear on latent pairs) and energy
                       (linear vs QUADRATIC: E ~ ||v||^2 so quadratic should win)
  3. stability      -- is the decoded energy flat WITHIN a trajectory?
  4. surprise       -- one-step prediction error by violation type
                       (continuity: teleport/phantom_bounce vs conservation:
                        energy_gain/energy_loss)

The eval set (bounce2d_eval.npz, paired normal+violated, seed 777) is generated
on first use with the speed range from --speed-min/--speed-max, which must match
the training distribution.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def epoch_of(path):
    m = re.search(r"e-(\d+)\.pth", path)
    return int(m.group(1)) if m else 10 ** 9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run folder containing e-*.pth.tar checkpoints")
    ap.add_argument("--checkpoint", default="latest.pth.tar",
                    help="checkpoint file name inside --run-dir (e.g. e-300.pth.tar)")
    ap.add_argument("--sweep", action="store_true",
                    help="also probe every e-*.pth.tar checkpoint in --run-dir")
    ap.add_argument("--max-epoch", type=int, default=None,
                    help="ignore sweep checkpoints beyond this epoch "
                         "(e.g. 300 for the published Exp A run, whose folder "
                         "also contains stale checkpoints from other runs)")
    ap.add_argument("--speed-min", type=float, default=0.02)
    ap.add_argument("--speed-max", type=float, default=0.10)
    ap.add_argument("--out", default=None,
                    help="output figure path (default: <run-dir>/probe_figures.png)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.chdir(REPO_ROOT)
    import probes as P

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- eval set (paired normal + 4 violations), regenerated if missing ---
    if not os.path.exists("bounce2d_eval.npz"):
        env = dict(os.environ)
        env["BOUNCE_SPEED_MIN"] = str(args.speed_min)
        env["BOUNCE_SPEED_MAX"] = str(args.speed_max)
        subprocess.run([sys.executable, "make_data.py"], cwd=REPO_ROOT, env=env, check=True)
    d = np.load("bounce2d_eval.npz", allow_pickle=True)
    eval_data = {"frames": d["frames"], "labels": d["labels"], "meta": d["meta"],
                 "gt": {k[3:]: d[k] for k in d.files if k.startswith("gt_")}}
    kinds = np.array([m[1] for m in eval_data["meta"]])
    norm = kinds == "normal"
    frames_t = torch.from_numpy(eval_data["frames"]).float()
    gt_n = {k: v[norm] for k, v in eval_data["gt"].items()}

    # --- detailed probes on the requested checkpoint ---
    ck = os.path.join(args.run_dir, args.checkpoint)
    print("checkpoint:", ck, "| exists:", os.path.exists(ck))
    model = P.load_checkpoint_model(ck, device=device)
    lat_n = P.extract_latents(model, frames_t, device=device)[norm]

    print("\n[Probe 1] instantaneous baseline (single frame)")
    for k, v in P.probe_instantaneous(lat_n, gt_n).items():
        print(f"  {k:<28}: {v:+.3f}")
    print("[Probe 2] energy: linear vs QUADRATIC (latent pairs)")
    r2 = P.probe_temporal(lat_n, gt_n)
    for k, v in r2.items():
        print(f"  {k:<28}: {v:+.3f}")
    print("[Probe 3] stability of the decoded invariant (bounces excluded)")
    for k, v in P.probe_stability(lat_n, gt_n, exclude_bounces=True).items():
        print(f"  {k:<28}: {v}")
    print("[Probe 4] surprise by violation type")
    res, _ = P.surprise_by_violation(model, eval_data, device=device)
    for k, v in res.items():
        print(f"  {k}: {v}")
    offs, curves = P.surprise_curve_by_violation(model, eval_data, device=device)

    # --- optional checkpoint sweep ---
    rows = []
    if args.sweep:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "e-*.pth.tar")), key=epoch_of)
        if args.max_epoch is not None:
            ckpts = [c for c in ckpts if epoch_of(c) <= args.max_epoch]
        print("\n=== SWEEP === checkpoints:", [os.path.basename(c) for c in ckpts])
        print(f"{'epoch':>7} | {'vel_R2':>7} | {'E_quad':>7} | {'within':>7} | {'across':>7}")
        for c in ckpts:
            ep = epoch_of(c)
            m = P.load_checkpoint_model(c, device=device)
            ln = P.extract_latents(m, frames_t, device=device)[norm]
            t = P.probe_temporal(ln, gt_n)
            s = P.probe_stability(ln, gt_n, exclude_bounces=True)
            rows.append((ep, t["velocity_R2_linear"], t["energy_R2_quadratic"]))
            print(f"{ep:>7} | {t['velocity_R2_linear']:>7.3f} | {t['energy_R2_quadratic']:>7.3f} | "
                  f"{s['within_traj_energy_CV_median']:>7.3f} | {s['across_traj_energy_CV']:>7.3f}")

    # --- figures ---
    ncols = 3 if rows else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5.3 * ncols, 4))
    ax[0].bar(["linear", "quadratic"],
              [r2["energy_R2_linear"], r2["energy_R2_quadratic"]], color=["#bbb", "#3b7"])
    ax[0].axhline(0, color="k", lw=.5)
    ax[0].set_title(f"energy linear vs quadratic ({os.path.basename(ck)})")
    ax[0].set_ylabel("R2")
    for k, c in curves.items():
        ax[1].plot(offs, c, "--" if k == "normal" else "-", marker="o", ms=3, label=k)
    ax[1].axvline(0, color="k", lw=.5)
    ax[1].legend(fontsize=7)
    ax[1].set_title("surprise curve (aligned at violation)")
    ax[1].set_xlabel("steps since violation")
    if rows:
        xs = [str(r[0]) for r in rows]
        ax[2].plot(xs, [r[1] for r in rows], "o-", label="velocity R2 (linear)")
        ax[2].plot(xs, [r[2] for r in rows], "s-", label="energy R2 (quadratic)")
        ax[2].axhline(0, color="k", lw=.5)
        ax[2].legend()
        ax[2].set_title("checkpoint sweep")
        ax[2].set_xlabel("epoch")
    plt.tight_layout()
    out = args.out or os.path.join(args.run_dir, "probe_figures.png")
    plt.savefig(out, dpi=120)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
