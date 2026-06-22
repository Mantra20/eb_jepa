"""Adapter: bounce2D numpy generator -> EB-JEPA AC-Video pipeline.

The AC-Video training loop unpacks a 5-tuple (x, a, loc, _, _) and calls
loader.dataset.normalizer.unnormalize_mse(...). Two Rooms uses SEPARATE train/val
datasets (no random_split), each exposing .normalizer -- we mirror that exactly.
"""
from typing import NamedTuple
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from bounce2d import build_dataset
from eb_jepa.datasets.two_rooms.normalizer import Normalizer


class Bounce2DNormalizer(Normalizer):
    """Reuse the repo Normalizer with bounce2D-appropriate stats.
    Locations live in box coords [0,1] (vs Two Rooms pixel coords); 2 channels in [0,1]."""
    def __init__(self):
        super().__init__()
        self.location_mean = torch.tensor([0.5, 0.5])
        self.location_std = torch.tensor([0.29, 0.29])      # ~ std of U[0,1]
        self.state_mean = torch.tensor([0.02, 0.06])        # ball sparse, walls border
        self.state_std = torch.tensor([0.12, 0.24])


class Bounce2DSample(NamedTuple):
    states:    torch.Tensor  # [2, T, S, S]  ch0=ball, ch1=walls
    actions:   torch.Tensor  # [2, T]        zeros (passive)   -- (A, T) order
    locations: torch.Tensor  # [2, T]        ball (x,y) GT
    energy:    torch.Tensor  # [T]           conserved invariant -- EVAL ONLY
    labels:    torch.Tensor  # [T]           violation id -- EVAL ONLY


class Bounce2DDataset(Dataset):
    def __init__(self, data):
        self.states = torch.from_numpy(np.ascontiguousarray(data["frames"])).float()  # [N,2,T,S,S]
        N, _, T, _, _ = self.states.shape
        self.actions = torch.zeros(N, 2, T, dtype=torch.float32)                       # [N,2,T]
        self.loc = torch.stack([torch.from_numpy(data["gt"]["x"]),
                                torch.from_numpy(data["gt"]["y"])], dim=1).float()      # [N,2,T]
        self.energy = torch.from_numpy(data["gt"]["E"]).float()                         # [N,T]
        self.labels = torch.from_numpy(data["labels"]).long()                          # [N,T]
        self.normalizer = Bounce2DNormalizer()

    def __len__(self): return self.states.shape[0]
    def get_seq_length(self, idx): return self.states.shape[2]

    def __getitem__(self, i):
        return Bounce2DSample(self.states[i], self.actions[i],
                              self.loc[i], self.energy[i], self.labels[i])


def build_bounce2d_loaders(cfg_data):
    """Return (train_loader, val_loader, data_config) for init_data dispatch.
    Training data is NORMAL-only (correct physics); violations are eval-only."""
    cfg = cfg_data or {}
    bs = int(cfg.get("batch_size", 384))
    T = int(cfg.get("T", 16))
    img = int(cfg.get("img_size", 65))
    n_base = int(cfg.get("n_base", 4000))
    n_val = int(cfg.get("n_val", 256))
    nw = int(cfg.get("num_workers", 0))
    pin = bool(cfg.get("pin_mem", False))
    persist = bool(cfg.get("persistent_workers", False)) and nw > 0
    sr = tuple(cfg.get("speed_range", (0.02, 0.06)))

    train = Bounce2DDataset(build_dataset(n_base=n_base, T=T, paired=False, S=img,
                                          speed_range=sr, seed0=0))
    val = Bounce2DDataset(build_dataset(n_base=n_val, T=T, paired=False, S=img,
                                        speed_range=sr, seed0=10_000_000))
    loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nw,
                        pin_memory=pin, drop_last=True, persistent_workers=persist)
    val_loader = DataLoader(val, batch_size=min(bs, len(val)), shuffle=False,
                            num_workers=nw, pin_memory=pin, drop_last=True,
                            persistent_workers=persist)
    data_config = SimpleNamespace(batch_size=bs, size=len(train), val_size=len(val),
                                  img_size=img)
    return loader, val_loader, data_config


@torch.no_grad()
def effective_rank(z):
    """Exponential entropy of the normalized covariance spectrum (Roy & Vetterli 2007).
    z: [N, D]. Collapse diagnostic independent of the std hinge. Log per epoch."""
    z = z - z.mean(0, keepdim=True)
    cov = (z.T @ z) / (z.shape[0] - 1)
    s = torch.linalg.svdvals(cov.float())
    p = s / (s.sum() + 1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
