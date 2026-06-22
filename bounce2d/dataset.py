"""Assemble numpy bundle. TRAINING set = normal only (paired=False): the world model
must learn CORRECT physics. Violations (paired=True) are for EVAL/probing only."""
import numpy as np
from .violations import generate_trajectory
from .render import render_trajectory

VIOLATION_KINDS = ["teleport", "phantom_bounce", "energy_gain", "energy_loss"]


def build_dataset(n_base=2000, T=16, paired=False, kinds=VIOLATION_KINDS,
                  box=1.0, radius=0.06, mass=1.0, speed_range=(0.02, 0.06),
                  t_violation_range=None, intensity=1.6, S=65, seed0=0):
    if t_violation_range is None:
        t_violation_range = (T // 4, 3 * T // 4)
    rng = np.random.default_rng(seed0)
    frames_list, gt_list, label_list, meta = [], [], [], []

    def add(rec, label, base_id, kind, t_v):
        frames_list.append(render_trajectory(rec, box, radius, S))
        gt_list.append({k: rec[k].astype(np.float32) for k in ("x","y","vx","vy","E","p")})
        label_list.append(label); meta.append((base_id, kind, t_v))

    for base_id in range(n_base):
        seed = seed0 + base_id
        t_v = int(rng.integers(*t_violation_range))
        rec, lab = generate_trajectory(seed, T, "normal", t_violation=t_v, box=box,
                                       radius=radius, mass=mass, speed_range=speed_range)
        add(rec, lab, base_id, "normal", t_v)
        if paired:
            for kind in kinds:
                rec, lab = generate_trajectory(seed, T, kind, t_violation=t_v,
                                               intensity=intensity, box=box, radius=radius,
                                               mass=mass, speed_range=speed_range)
                add(rec, lab, base_id, kind, t_v)

    frames = np.stack(frames_list).astype(np.float32)         # [N,2,T,S,S]
    gt = {k: np.stack([g[k] for g in gt_list]) for k in gt_list[0]}
    labels = np.stack(label_list).astype(np.int64)
    meta = np.array(meta, dtype=object)
    return dict(frames=frames, gt=gt, labels=labels, meta=meta)
