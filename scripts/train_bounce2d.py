"""Launch an AC-Video-JEPA training run on bounce2D.

Thin, reproducible wrapper around examples/ac_video_jepa/main.py with the exact
hyper-parameters used for the two published runs:

    Exp A (main):      python scripts/train_bounce2d.py --exp-name main_simt12     --sim-coeff 12 --epochs 300
    Exp B (ablation):  python scripts/train_bounce2d.py --exp-name ablation_simt0  --sim-coeff 0  --epochs 100

The ONLY variable manipulated between Exp A and Exp B is `sim_coeff_t` (the
temporal-similarity regularization coefficient) -- everything else is constant.
Checkpoints are written to <out-dir>/<exp-name>_seed<seed>/ every 10 epochs
(e-{epoch}.pth.tar) plus latest.pth.tar, so a post-hoc checkpoint sweep never
requires retraining.

The speed range is propagated through BOUNCE_SPEED_MIN/MAX environment
variables so that training data and the held-out eval set are generated with
the same distribution. (0.02, 0.10) gives a x25 spread on v^2, i.e. enough
across-trajectory energy variance (CV ~ 0.70) for the probes to have signal.

Watch the per-epoch logs for collapse: eff_rank -> 1-2 means representational
collapse, temp_var -> 0 means temporal collapse. Healthy: eff_rank > 5,
temp_var > 0.1.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-name", default="main_simt12",
                    help="experiment name; run dir is <out-dir>/<exp-name>_seed<seed>")
    ap.add_argument("--sim-coeff", type=float, default=12,
                    help="sim_coeff_t: 12 = Exp A (main) | 0 = Exp B (ablation)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-base", type=int, default=4000,
                    help="number of training trajectories (~4.3 GB RAM at 4000)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--speed-min", type=float, default=0.02)
    ap.add_argument("--speed-max", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "runs"),
                    help="parent folder for run dirs (default: <repo>/runs)")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    run_dir = os.path.join(args.out_dir, f"{args.exp_name}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    env = dict(os.environ)
    env["BOUNCE_SPEED_MIN"] = str(args.speed_min)
    env["BOUNCE_SPEED_MAX"] = str(args.speed_max)

    cmd = [sys.executable, "-m", "examples.ac_video_jepa.main",
           "--fname=examples/ac_video_jepa/cfgs/train.yaml",
           f"--meta.model_folder={run_dir}",
           f"--meta.seed={args.seed}",
           "--logging.save_every_n_epochs=10",
           "--data.env_name=bounce2d", "--data.img_size=65", "--data.T=16",
           f"--data.n_base={args.n_base}",
           f"--data.batch_size={args.batch_size}",
           f"--data.num_workers={args.num_workers}",
           "--data.pin_mem=False", "--data.persistent_workers=False",
           "--model.dobs=2", "--model.nsteps=8",
           "--model.regularizer.std_margin=2.0",
           "--model.regularizer.cov_coeff=12", "--model.regularizer.std_coeff=16",
           f"--model.regularizer.sim_coeff_t={args.sim_coeff}",
           "--model.regularizer.idm_coeff=0",
           "--training.dtype=bfloat16", "--training.use_amp=True",
           "--model.compile=False",
           "--meta.enable_plan_eval=False", "--meta.load_model=False",
           "--logging.log_wandb=False",
           f"--optim.epochs={args.epochs}"]

    print(f"RUN: {run_dir} | sim_coeff_t={args.sim_coeff} | n_base={args.n_base} "
          f"| epochs={args.epochs} | speed=({args.speed_min},{args.speed_max})")
    print("Monitor eff_rank (->1-2 = collapse) and temp_var (->0 = temporal collapse)\n")

    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        sys.stdout.flush()
    proc.wait()
    print("\nreturncode:", proc.returncode, "(-9 = OOM: lower --n-base)")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
