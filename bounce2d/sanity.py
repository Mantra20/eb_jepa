"""Generation-time correctness checks. Run BEFORE training. Fails loudly.
Works on a normal-only set (format/energy/mass) and a paired set (adds orthogonality/pairing)."""
import numpy as np


def check_dataset(data, verbose=True):
    gt, meta, frames = data["gt"], data["meta"], data["frames"]
    E = gt["E"]
    kinds = np.array([m[1] for m in meta]); t_vs = np.array([m[2] for m in meta])
    is_normal = kinds == "normal"; report = {}

    assert frames.ndim == 5 and frames.shape[1] == 2 and frames.shape[-2] == frames.shape[-1]
    assert frames.dtype == np.float32 and frames.min() >= 0 and frames.max() <= 1
    report["format"] = tuple(frames.shape)

    soft = ((frames[:, 0] > 0.05) & (frames[:, 0] < 0.95)).sum()
    assert soft > 0, "ball channel binary -> centroid risk"; report["soft_pixels"] = int(soft)

    intra = E[is_normal].std(axis=1); report["max_intra_std_normal"] = float(intra.max())
    assert intra.max() < 1e-10, "energy not conserved intra-traj"

    inter = E[is_normal].mean(axis=1).std(); report["inter_traj_energy_std"] = float(inter)
    assert inter > 0, "energy is a dataset constant -> probe proves nothing"

    speed = np.sqrt(gt["vx"] ** 2 + gt["vy"] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_est = np.where(speed > 0, gt["p"] / speed, np.nan)
    m_vals = m_est[np.isfinite(m_est)]
    report["mass_spread"] = float(np.nanmax(m_vals) - np.nanmin(m_vals))
    assert report["mass_spread"] < 1e-4, "mass not fixed -> identifiability problem"

    for kind in ("teleport", "phantom_bounce", "energy_gain", "energy_loss"):
        idx = np.where(kinds == kind)[0]
        if len(idx) == 0:
            continue  # normal-only set: nothing to check for this kind
        rel_dE, jr = [], []
        for i in idx:
            tv = t_vs[i]
            rel_dE.append(abs(E[i, tv] - E[i, tv - 1]) / E[i, 0])
            disp = np.sqrt(np.diff(gt["x"][i]) ** 2 + np.diff(gt["y"][i]) ** 2)
            jr.append(disp.max() / (np.median(disp) + 1e-12))
        rel_dE, jr = float(np.mean(rel_dE)), float(np.mean(jr))
        report[f"{kind}_dE"], report[f"{kind}_jump"] = rel_dE, jr
        if kind in ("teleport", "phantom_bounce"):
            assert rel_dE < 1e-6 and jr > 5, f"{kind} not clean continuity-only"
        else:
            assert rel_dE > 1e-2 and jr < 3, f"{kind} not clean conservation-only"

    if not is_normal.all():
        bids = np.array([m[0] for m in meta])
        for b in np.unique(bids)[:5]:
            ni = np.where((bids == b) & (kinds == "normal"))[0][0]
            for vi in np.where((bids == b) & (kinds != "normal"))[0]:
                tv = t_vs[vi]
                assert np.allclose(gt["x"][ni, :tv], gt["x"][vi, :tv]), "pair prefix mismatch"
        report["pairing"] = "ok"

    if verbose:
        for k, v in report.items(): print(f"  {k:<28}: {v}")
    return report
