"""Render a trajectory to [2, T, S, S], VECTORIZED over time (no python per-frame loop).
Ch0 = anti-aliased ball (soft edges, not binary -> avoids trivial centroid encoder).
Ch1 = static wall mask. 2 channels == ImpalaEncoder input_channels (dobs=2).
Channels stay purely visual (no velocity/energy written -> no leakage)."""
import numpy as np


def render_trajectory(rec, box, radius, S=65, aa=1.2):
    x = np.asarray(rec["x"], dtype=np.float64)
    y = np.asarray(rec["y"], dtype=np.float64)
    T = x.shape[0]
    g = (np.arange(S) + 0.5) / S * box
    XX, YY = np.meshgrid(g, g)                                  # [S,S]
    edge = aa / (S / box)
    dx = XX[None] - x[:, None, None]                            # [T,S,S] (broadcast)
    dy = YY[None] - y[:, None, None]
    dist = np.sqrt(dx * dx + dy * dy)
    ball = np.clip((radius - dist) / edge + 0.5, 0.0, 1.0).astype(np.float32)
    walls = np.zeros((S, S), dtype=np.float32)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = 1.0
    walls = np.broadcast_to(walls, (T, S, S))
    return np.stack([ball, walls], axis=0)                      # [2,T,S,S]
