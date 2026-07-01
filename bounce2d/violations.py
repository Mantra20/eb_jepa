"""Trajectory generation with optional violation at t_violation. 4 types, ORTHOGONAL:
  CONTINUITY (energy preserved): teleport (pos jump), phantom_bounce (through wall)
  CONSERVATION (pos continuous): energy_gain (|v|*=k), energy_loss (|v|/=k)
Pairing: same seed for normal & a violation -> identical prefix up to t_violation."""
import numpy as np
from .env import BounceEnv

KIND_ID = {"normal": 0, "teleport": 1, "phantom_bounce": 2,
           "energy_gain": 3, "energy_loss": 4}


def generate_trajectory(seed, T, kind="normal", t_violation=None, intensity=1.6,
                        box=1.0, radius=0.06, mass=1.0, speed_range=(0.02, 0.10)):
    env = BounceEnv(box=box, radius=radius, mass=mass,
                    speed_range=speed_range, rng=np.random.default_rng(seed))
    env.reset()
    keys = ("x", "y", "vx", "vy", "E", "p")
    rec = {k: np.empty(T) for k in keys}
    label = np.zeros(T, dtype=np.int64)
    phantom_done = False
    for t in range(T):
        if t > 0:
            env.x += env.vx * env.dt; env.y += env.vy * env.dt
            lo, hi = env.r, env.box - env.r
            hit = (env.x < lo) or (env.x > hi) or (env.y < lo) or (env.y > hi)
            if kind == "phantom_bounce" and t >= (t_violation or 0) and hit and not phantom_done:
                if env.x < lo:   env.x = hi - (lo - env.x)
                elif env.x > hi: env.x = lo + (env.x - hi)
                if env.y < lo:   env.y = hi - (lo - env.y)
                elif env.y > hi: env.y = lo + (env.y - hi)
                phantom_done = True
            else:
                env.x, env.vx = env._reflect(env.x, env.vx)
                env.y, env.vy = env._reflect(env.y, env.vy)
        if t_violation is not None and t == t_violation:
            if kind == "teleport":      env.x, env.y = env.box - env.x, env.box - env.y
            elif kind == "energy_gain": env.vx *= intensity; env.vy *= intensity
            elif kind == "energy_loss": env.vx /= intensity; env.vy /= intensity
        s = env.state()
        for k in keys: rec[k][t] = s[k]
        if kind != "normal" and t_violation is not None and t >= t_violation:
            label[t] = KIND_ID[kind]
    return rec, label
