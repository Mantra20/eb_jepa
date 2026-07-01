"""bounce2D physics: walled elastic box, single ball, fixed mass.
Energy conserved bit-exactly across reflections (sign flip preserves the square).
ICs randomized -> energy varies across trajectories, constant within each.
Momentum VECTOR not conserved in a walled box; probe ENERGY (and |v|)."""
import numpy as np


class BounceEnv:
    def __init__(self, box=1.0, radius=0.06, mass=1.0, dt=1.0,
                 speed_range=(0.02, 0.10), rng=None):
        self.box, self.r, self.m, self.dt = box, radius, mass, dt
        self.speed_range = speed_range
        self.rng = rng if rng is not None else np.random.default_rng()

    def reset(self):
        lo, hi = self.r, self.box - self.r
        self.x = self.rng.uniform(lo, hi); self.y = self.rng.uniform(lo, hi)
        s = self.rng.uniform(*self.speed_range); a = self.rng.uniform(0.0, 2 * np.pi)
        self.vx, self.vy = s * np.cos(a), s * np.sin(a)
        return self.state()

    def _reflect(self, p, v):
        lo, hi = self.r, self.box - self.r
        for _ in range(4):
            if p < lo:   p, v = 2 * lo - p, -v
            elif p > hi: p, v = 2 * hi - p, -v
            else: break
        return p, v

    def state(self):
        s2 = self.vx ** 2 + self.vy ** 2
        return dict(x=self.x, y=self.y, vx=self.vx, vy=self.vy,
                    E=0.5 * self.m * s2, p=self.m * np.sqrt(s2))
