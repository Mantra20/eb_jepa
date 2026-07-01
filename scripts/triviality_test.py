"""Triviality test: is energy encoded BEYOND what velocity alone explains?

Complements the official per-frame probe (probes.probe_temporal, pairs of
PCA-reduced latents) with a DIFFERENT question at the TRAJECTORY level
(time-averaged raw latent): can energy be decoded beyond what a LINEAR
reconstruction of speed would allow?

    python scripts/triviality_test.py --run-dir runs/main_simt12_seed1 --checkpoint e-300.pth.tar

Three probes on the same time-averaged latent X = mean_t z_t:
  1. direct quadratic: latent -> E        (PolynomialFeatures(2) + RidgeCV)
  2. indirect:         latent -> speed (linear), then E_hat = 1/2 speed_hat^2
  3. direct linear:    latent -> E        (baseline)

Non-triviality criterion: (1) clearly beating (2) means energy is encoded
beyond a linear read-out of speed. A permutation control (shuffled E) guards
against the direct quadratic probe's over-parameterization (~131k quadratic
features for ~640 train trajectories).

Methodological notes baked into this protocol (see docs/probing_energy_triviality.md):
  * predict the scalar speed mean, NOT the mean velocity vector -- averaging the
    vector over time suffers a Jensen bias (bounces cancel vx/vy components), so
    ||mean_t v_t||^2 != mean_t ||v_t||^2 and the indirect probe would be
    unfairly handicapped;
  * RidgeCV over alphas in [1e-6, 1e2] -- E_mean has tiny ABSOLUTE scale
    (~1e-3, variance ~1e-6), so sklearn's default alpha=1.0 over-regularizes
    massively regardless of signal quality (relative variation is fine:
    CV(E_mean) ~ 0.70).

IMPORTANT: this test and probes.probe_temporal measure DIFFERENT things
(trajectory-level raw latent here vs per-frame PCA(24) pairs there). Their R2
values are complementary, not replicates of one measurement.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="latest.pth.tar")
    ap.add_argument("--speed-min", type=float, default=0.02)
    ap.add_argument("--speed-max", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.utils import shuffle

    os.chdir(REPO_ROOT)
    import probes as P

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("bounce2d_eval.npz"):
        env = dict(os.environ)
        env["BOUNCE_SPEED_MIN"] = str(args.speed_min)
        env["BOUNCE_SPEED_MAX"] = str(args.speed_max)
        subprocess.run([sys.executable, "make_data.py"], cwd=REPO_ROOT, env=env, check=True)
    d = np.load("bounce2d_eval.npz", allow_pickle=True)
    gt = {k[3:]: d[k] for k in d.files if k.startswith("gt_")}
    kinds = np.array([m[1] for m in d["meta"]])
    norm = kinds == "normal"
    frames_t = torch.from_numpy(d["frames"]).float()
    gt_n = {k: v[norm] for k, v in gt.items()}

    ck = os.path.join(args.run_dir, args.checkpoint)
    print("checkpoint:", ck, "| exists:", os.path.exists(ck))
    model = P.load_checkpoint_model(ck, device=device)
    lat_n = P.extract_latents(model, frames_t, device=device)[norm]   # (N, T, D)

    # ---- targets ----
    X = lat_n.mean(axis=1)                                 # (N, D) time-averaged latent
    speed_t = np.sqrt(gt_n["vx"] ** 2 + gt_n["vy"] ** 2)   # (N, T)
    v_mean = np.stack([gt_n["vx"], gt_n["vy"]], -1).mean(axis=1)  # (N, 2) Jensen-biased
    speed_mean = speed_t.mean(axis=1)                      # (N,)  correct target
    E_mean = gt_n["E"].mean(axis=1)                        # (N,)

    # Sanity: speed_mean reconstructs E exactly; the mean VECTOR does not (Jensen bias).
    ss_tot = ((E_mean - E_mean.mean()) ** 2).sum()
    r2_speed = 1 - ((E_mean - 0.5 * speed_mean ** 2) ** 2).sum() / ss_tot
    r2_vec = 1 - ((E_mean - 0.5 * (v_mean ** 2).sum(axis=1)) ** 2).sum() / ss_tot
    print(f"[sanity] R2(E, 0.5*speed_mean^2)  = {r2_speed:.3f}  (expected ~1.0)")
    print(f"[sanity] R2(E, 0.5*||v_mean||^2)  = {r2_vec:.3f}  (expected bad: Jensen bias)")

    alphas = np.logspace(-6, 2, 25)

    # ---- 1: direct quadratic probe latent -> E ----
    pipe_direct = make_pipeline(StandardScaler(),
                                PolynomialFeatures(degree=2, include_bias=False),
                                RidgeCV(alphas=alphas))
    r2_direct = cross_val_score(pipe_direct, X, E_mean, cv=5, scoring="r2").mean()
    print(f"\ndirect quadratic  latent -> E                : R2={r2_direct:.3f}")

    # ---- 2: indirect: linear speed_hat, then E_hat = 1/2 speed_hat^2 ----
    pipe_speed = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(X):
        pipe_speed.fit(X[tr], speed_mean[tr])
        E_hat = 0.5 * pipe_speed.predict(X[te]) ** 2
        ss_res = ((E_mean[te] - E_hat) ** 2).sum()
        sst = ((E_mean[te] - E_mean[te].mean()) ** 2).sum()
        scores.append(1 - ss_res / sst)
    r2_indirect = float(np.mean(scores))
    print(f"indirect  latent -> speed -> E=1/2 speed^2   : R2={r2_indirect:.3f}")

    # ---- 3: direct linear baseline ----
    pipe_lin = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    r2_linear = cross_val_score(pipe_lin, X, E_mean, cv=5, scoring="r2").mean()
    print(f"direct linear     latent -> E                : R2={r2_linear:.3f}")

    # ---- verdict ----
    delta = r2_direct - r2_indirect
    print(f"\n{'=' * 50}\ndelta (direct - indirect): {delta:+.3f}")
    if delta > 0.05:
        print("-> E is encoded BEYOND v: non-trivial result")
    elif delta > 0:
        print("-> slight surplus, interpret with caution")
    else:
        print("-> E is essentially derived from v: trivial result")

    # ---- permutation control ----
    r2_perm = [cross_val_score(pipe_direct, X, shuffle(E_mean), cv=5, scoring="r2").mean()
               for _ in range(5)]
    print(f"\n[control] R2 on shuffled E (should be ~0 or negative): "
          f"{np.mean(r2_perm):.3f} +/- {np.std(r2_perm):.3f}")
    if np.mean(r2_perm) < r2_direct - 0.2:
        print("[control] gap between real and shuffled R2 confirms the signal "
              "is not a dimensionality artifact.")
    else:
        print("[control] WARNING: shuffled R2 close to real R2 -- the signal may "
              "be an over-parameterization artifact.")


if __name__ == "__main__":
    main()
