"""
Probing of bounce2D JEPA latents — does the world model encode a CONSERVED
INVARIANT (kinetic energy), and is it "surprised" by its violation?

----------------------------------------------------------------------------
WHY THESE PROBES, AND WHY LINEAR *AND* QUADRATIC
----------------------------------------------------------------------------
The AC-Video encoder is applied frame-by-frame: it maps each frame to a global
latent z_t (the predictor handles time separately). Consequences that dictate
the probe design:

  (1) A SINGLE frame contains POSITION but NOT velocity (one snapshot has no
      motion). So an instantaneous probe of z_t can recover (x, y) but must
      FAIL on velocity. This is the baseline that motivates everything else.

  (2) Velocity is observable only across >=2 frames. From a pair
      g_t = [z_{t-1}, z_t], velocity ~ (pos_t - pos_{t-1}) is a LINEAR function
      of g_t (because position is linear in z). Hence a LINEAR probe on pairs
      should recover velocity.

  (3) KINETIC ENERGY IS QUADRATIC: E = 1/2 m||v||^2 = 1/2 m (vx^2 + vy^2).
      Even if the latent perfectly encodes the velocity VECTOR, energy is a
      QUADRATIC function of it, so a LINEAR probe for energy can fail while a
      QUADRATIC probe succeeds. Testing linear-vs-quadratic decodability is the
      physically correct way to probe a quadratic invariant: if quadratic >>
      linear, the representation encodes the underlying vector state (velocity)
      and energy is a quadratic read-out of it — not stored as a scalar.

A quantity is only worth probing if it VARIES in the data: energy is constant
within a trajectory (conservation) but varies ACROSS trajectories thanks to
randomized initial conditions. The probes regress across trajectories.

No-leakage: probes see ONLY the latents (frames -> encoder). Ground-truth
positions/velocities/energy are used solely as regression TARGETS, offline.
Train/test split is BY TRAJECTORY so frames of one trajectory never straddle
the split.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Model (re)construction + latent extraction
# ---------------------------------------------------------------------------
def build_model(dobs=2, img=65, henc=32, dstc=32, device="cpu"):
    """Rebuild the AC-Video JEPA exactly as examples/ac_video_jepa/main.py."""
    from eb_jepa.architectures import ImpalaEncoder, InverseDynamicsModel, RNNPredictor
    from eb_jepa.jepa import JEPA
    from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
    enc = ImpalaEncoder(width=1, stack_sizes=(16, henc, dstc), num_blocks=2,
                        dropout_rate=None, layer_norm=False, input_channels=dobs,
                        final_ln=True, mlp_output_dim=512, input_shape=(dobs, img, img))
    test = enc(torch.rand(1, dobs, 1, img, img))
    _, f, _, h, w = test.shape
    pred = RNNPredictor(hidden_size=enc.mlp_output_dim, final_ln=enc.final_ln)
    idm = InverseDynamicsModel(state_dim=h * w * f, hidden_dim=256, action_dim=2)
    reg = VC_IDM_Sim_Regularizer(cov_coeff=12, std_coeff=16, sim_coeff_t=12,
                                 idm_coeff=1, std_margin=2.0, idm=idm,
                                 first_t_only=False, projector=None,
                                 spatial_as_samples=False, idm_after_proj=False,
                                 sim_t_after_proj=False)
    jepa = JEPA(enc, nn.Identity(), pred, reg, SquareLossSeq()).to(device)
    return jepa


def load_checkpoint_model(ckpt_path, device="cpu", **kw):
    """Rebuild the model and load a trained checkpoint's model_state_dict."""
    jepa = build_model(device=device, **kw)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    jepa.load_state_dict(ck["model_state_dict"], strict=False)
    jepa.eval()
    return jepa


@torch.no_grad()
def extract_latents(model, states, batch=128, device="cpu"):
    """states [N,2,T,H,W] -> per-frame encoder latents [N, T, D]."""
    outs = []
    for i in range(0, states.shape[0], batch):
        x = states[i:i + batch].to(device)
        z = model.encoder(x)                       # [b, D, T, 1, 1]
        outs.append(z.squeeze(-1).squeeze(-1).permute(0, 2, 1).cpu())  # [b, T, D]
    return torch.cat(outs).numpy()


# ---------------------------------------------------------------------------
# Regression helpers (closed-form ridge; standardized inputs; split by traj)
# ---------------------------------------------------------------------------
def _split_by_traj(n, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def _r2_multi(model, Xtr, Ytr, Xte, Yte):
    """Fit a regressor on train, return mean R^2 over targets on test."""
    model.fit(Xtr, Ytr)
    pred = model.predict(Xte)
    if pred.ndim == 1:
        pred = pred[:, None]
    if Yte.ndim == 1:
        Yte = Yte[:, None]
    return float(np.mean([r2_score(Yte[:, j], pred[:, j]) for j in range(Yte.shape[1])]))


def linear_probe(X, Y, ntraj, T, alpha=1.0, seed=0):
    """Standardize -> Ridge. X is [N*T, F], Y is [N*T, k]. Split BY TRAJECTORY."""
    tr, te = _split_by_traj(ntraj, seed=seed)
    fr = lambda idx: np.concatenate([np.arange(i * T, (i + 1) * T) for i in idx])
    itr, ite = fr(tr), fr(te)
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return _r2_multi(pipe, X[itr], Y[itr], X[ite], Y[ite])


def quadratic_probe(X, Y, ntraj, T, k=24, alpha=1.0, seed=0):
    """PCA(k) -> degree-2 polynomial features (linear + squares + cross) -> Ridge.
    The degree-2 expansion is exactly what lets the probe represent E ~ ||v||^2."""
    tr, te = _split_by_traj(ntraj, seed=seed)
    fr = lambda idx: np.concatenate([np.arange(i * T, (i + 1) * T) for i in idx])
    itr, ite = fr(tr), fr(te)
    k = min(k, X.shape[1])
    pipe = make_pipeline(StandardScaler(), PCA(n_components=k),
                         PolynomialFeatures(degree=2, include_bias=False),
                         StandardScaler(), Ridge(alpha=alpha))
    return _r2_multi(pipe, X[itr], Y[itr], X[ite], Y[ite])


# ---------------------------------------------------------------------------
# PROBE 1 — instantaneous (single frame). BASELINE (H1).
# ---------------------------------------------------------------------------
def probe_instantaneous(latents, gt):
    """Single-frame latent z_t -> position and velocity.

    EXPECTATION: position R^2 high (a frame shows where the ball is);
    velocity R^2 ~ 0 (a single frame has no motion). The velocity floor here
    is the baseline the temporal probes must beat — it shows energy CANNOT be
    read from one frame, so any later energy signal is genuinely temporal.
    """
    N, T, D = latents.shape
    Z = latents.reshape(N * T, D)
    pos = np.stack([gt["x"], gt["y"]], -1).reshape(N * T, 2)
    vel = np.stack([gt["vx"], gt["vy"]], -1).reshape(N * T, 2)
    return {
        "position_R2_linear": linear_probe(Z, pos, N, T),
        "velocity_R2_linear": linear_probe(Z, vel, N, T),  # expected ~0
    }


# ---------------------------------------------------------------------------
# PROBE 2 — temporal pairs. velocity (linear) + energy (LINEAR vs QUADRATIC).
# ---------------------------------------------------------------------------
def probe_temporal(latents, gt):
    """Pair g_t = [z_{t-1}, z_t] -> velocity, speed, and ENERGY.

    velocity / |v|: LINEAR probe (motion is a linear function of the pair).
    energy: LINEAR probe vs QUADRATIC probe on the SAME PCA base. Because
    E ~ ||v||^2, the quadratic probe should clearly beat the linear one. That
    gap is the evidence that the latent holds the velocity VECTOR and energy is
    a quadratic read-out — exactly what a physically faithful representation
    should look like.
    """
    N, T, D = latents.shape
    g = np.concatenate([latents[:, :-1, :], latents[:, 1:, :]], -1)  # [N, T-1, 2D]
    Tm = T - 1
    G = g.reshape(N * Tm, 2 * D)
    vx, vy = gt["vx"][:, 1:], gt["vy"][:, 1:]
    vel = np.stack([vx, vy], -1).reshape(N * Tm, 2)
    speed = np.sqrt(vx ** 2 + vy ** 2).reshape(N * Tm, 1)
    energy = gt["E"][:, 1:].reshape(N * Tm, 1)
    return {
        "velocity_R2_linear": linear_probe(G, vel, N, Tm),
        "speed_R2_linear":    linear_probe(G, speed, N, Tm),
        "energy_R2_linear":   linear_probe(G, energy, N, Tm),      # baseline
        "energy_R2_quadratic": quadratic_probe(G, energy, N, Tm),   # should win
    }


# ---------------------------------------------------------------------------
# PROBE 3 — temporal stability of the invariant (H2).
# ---------------------------------------------------------------------------
def _bounce_mask(gt):
    """True where a bounce occurred in transition (t-1 -> t): a velocity
    component flipped sign. Shape [N, T-1] (aligned with pair index)."""
    vx, vy = gt["vx"], gt["vy"]
    return (vx[:, 1:] * vx[:, :-1] < 0) | (vy[:, 1:] * vy[:, :-1] < 0)


def probe_stability(latents, gt, k=24, alpha=1.0, exclude_bounces=True):
    """Decode energy at every t (quadratic head fit across trajectories), then
    measure how FLAT the decoded energy is WITHIN each trajectory.

    True energy is constant in time, so a faithful representation yields a
    decoded energy Ê_t that is nearly flat. We report the median within-traj
    coefficient of variation of Ê_t (std_t / |mean_t|). Lower = the invariant is
    held stable over time, not merely decodable at isolated instants. The
    across-trajectory CV is reported as context (a meaningful within-CV must be
    << across-CV).

    exclude_bounces=True drops transitions containing a wall bounce: there the
    pair displacement != velocity, so the pair-based energy estimate is noisy by
    construction. Excluding them removes a known noise floor and isolates the
    representation's own stability.
    """
    N, T, D = latents.shape
    g = np.concatenate([latents[:, :-1, :], latents[:, 1:, :]], -1)
    Tm = T - 1
    G = g.reshape(N * Tm, 2 * D)
    E = gt["E"][:, 1:].reshape(N * Tm, 1)
    tr, te = _split_by_traj(N, seed=0)
    fr = lambda idx: np.concatenate([np.arange(i * Tm, (i + 1) * Tm) for i in idx])
    itr, ite = fr(tr), fr(te)
    k = min(k, G.shape[1])
    pipe = make_pipeline(StandardScaler(), PCA(n_components=k),
                         PolynomialFeatures(degree=2, include_bias=False),
                         StandardScaler(), Ridge(alpha=alpha))
    pipe.fit(G[itr], E[itr].ravel())
    Ehat = pipe.predict(G[ite]).reshape(len(te), Tm)
    if exclude_bounces:
        bm = _bounce_mask(gt)[te]                       # [n_test, Tm]
        Ehat = np.where(bm, np.nan, Ehat)
        within = np.nanstd(Ehat, axis=1) / (np.abs(np.nanmean(Ehat, axis=1)) + 1e-9)
        across = np.nanstd(np.nanmean(Ehat, axis=1)) / (np.abs(np.nanmean(Ehat)) + 1e-9)
    else:
        within = np.std(Ehat, axis=1) / (np.abs(np.mean(Ehat, axis=1)) + 1e-9)
        across = np.std(np.mean(Ehat, axis=1)) / (np.abs(np.mean(Ehat)) + 1e-9)
    return {
        "within_traj_energy_CV_median": float(np.nanmedian(within)),
        "across_traj_energy_CV": float(across),
        "bounce_steps_excluded": bool(exclude_bounces),
    }


# ---------------------------------------------------------------------------
# PROBE 4 — surprise by violation type (H3/H4). THE HEADLINE.
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_surprise(model, states, actions, device="cpu", batch=128):
    """One-step latent prediction error e_t = ||predictor(z_{t-1}, a_{t-1}) - z_t||^2.

    The predictor was trained on NORMAL physics to map z_{t-1} -> z_t. A
    transition that violates the learned dynamics inflates e_t at that step.
    Teacher-forcing the context (predict from the TRUE previous latent) isolates
    per-step surprise rather than letting errors compound.
    """
    outs = []
    for i in range(0, states.shape[0], batch):
        x = states[i:i + batch].to(device)
        a = actions[i:i + batch].to(device)
        z = model.encoder(x)                                  # [b, D, T, 1, 1]
        T = z.shape[2]
        errs = []
        for t in range(1, T):
            state = z[:, :, t - 1:t]                           # [b, D, 1, 1, 1]
            act = a[:, :, t - 1:t]                             # [b, A, 1]
            pred = model.predictor(state, act)                # [b, D, 1, 1, 1]
            e = ((pred - z[:, :, t:t + 1]) ** 2).flatten(1).mean(1)  # [b]
            errs.append(e.cpu())
        outs.append(torch.stack(errs, 1))                     # [b, T-1]
    return torch.cat(outs).numpy()                            # [N, T-1]


def surprise_by_violation(model, eval_data, device="cpu", window=2):
    """Aggregate one-step surprise BY VIOLATION TYPE, aligned at t_violation.

    Orthogonality of the dataset is what makes this interpretable:
      - CONTINUITY violations (teleport, phantom_bounce) create a position jump
        -> any model that tracks position will be surprised. This is the
        expected-surprise reference.
      - CONSERVATION violations (energy_gain/loss) keep the position continuous;
        only the speed changes. A model that tracks ONLY position should NOT be
        surprised here. Surprise on these, ABOVE the continuity baseline, is the
        evidence that the model encodes the conserved invariant (H3/H4).
    """
    frames = torch.from_numpy(eval_data["frames"]).float()
    N = frames.shape[0]
    actions = torch.zeros(N, 2, frames.shape[2])
    meta = eval_data["meta"]
    kinds = np.array([m[1] for m in meta])
    tvs = np.array([int(m[2]) for m in meta])
    S = compute_surprise(model, frames, actions, device=device)   # [N, T-1]

    res = {}
    for kind in ["teleport", "phantom_bounce", "energy_gain", "energy_loss"]:
        idx = np.where(kinds == kind)[0]
        if len(idx) == 0:
            continue
        post, pre = [], []
        for i in idx:
            tv = tvs[i]
            a0 = max(tv - 1, 0); a1 = min(tv - 1 + window, S.shape[1])
            if a1 <= a0:
                continue
            post.append(S[i, a0:a1].mean())
            pre.append(S[i, :max(tv - 1, 1)].mean())
        post, pre = np.array(post), np.array(pre)
        res[kind] = {
            "surprise_post": float(post.mean()),
            "surprise_pre": float(pre.mean()),
            "ratio_post_over_pre": float(post.mean() / (pre.mean() + 1e-12)),
        }
    cons = [res[k]["surprise_post"] for k in ("energy_gain", "energy_loss") if k in res]
    cont = [res[k]["surprise_post"] for k in ("teleport", "phantom_bounce") if k in res]
    if cons and cont:
        res["_conservation_vs_continuity"] = float(np.mean(cons) / (np.mean(cont) + 1e-12))
    return res, S


def surprise_curve_by_violation(model, eval_data, pre=3, post=6, device="cpu"):
    """Mean one-step surprise CURVE, aligned at t_violation, per violation type.

    This is what actually separates the two violation families — the SHAPE, not
    a single number:
      - CONTINUITY (teleport, phantom_bounce): a transient spike at the violation
        transition, then recovery (velocity is unchanged, so prediction
        re-synchronizes the next step).
      - CONSERVATION (energy_gain/loss): a STEP that persists, because the speed
        changed for good and every subsequent one-step prediction (built on the
        old speed) stays off.
    'normal' trajectories aligned at the same t give a flat baseline.
    Returns (offsets, {kind: mean_curve}). offsets are relative to the violation.
    """
    frames = torch.from_numpy(eval_data["frames"]).float()
    N = frames.shape[0]
    actions = torch.zeros(N, 2, frames.shape[2])
    meta = eval_data["meta"]
    kinds = np.array([m[1] for m in meta]); tvs = np.array([int(m[2]) for m in meta])
    S = compute_surprise(model, frames, actions, device=device)   # [N, T-1]
    Tm = S.shape[1]
    offsets = np.arange(-pre, post + 1)
    curves = {}
    for kind in ["normal", "teleport", "phantom_bounce", "energy_gain", "energy_loss"]:
        idx = np.where(kinds == kind)[0]
        if len(idx) == 0:
            continue
        rows = []
        for i in idx:
            cols = (tvs[i] - 1) + offsets       # tv-1 = transition where violation first acts
            row = np.full(len(offsets), np.nan)
            valid = (cols >= 0) & (cols < Tm)
            row[valid] = S[i, cols[valid]]
            rows.append(row)
        curves[kind] = np.nanmean(np.stack(rows), axis=0)
    return offsets, curves
