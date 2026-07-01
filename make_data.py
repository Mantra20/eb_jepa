"""Generate bounce2D datasets and run sanity checks.
  - bounce2d_eval.npz  : PAIRED (normal + 4 violations) -> probes/surprise; sanity runs here.
  - (training data is generated on the fly by build_bounce2d_loaders, normal-only.)
Adjust n_base for training steps: more trajectories or more epochs = more steps."""
import numpy as np
from bounce2d import build_dataset, check_dataset

if __name__ == "__main__":
    print("=== EVAL set (paired: normal + 4 violations) ===")
    import os
    sr = (float(os.environ.get("BOUNCE_SPEED_MIN", 0.02)), float(os.environ.get("BOUNCE_SPEED_MAX", 0.10)))
    print("speed_range:", sr)
    ev = build_dataset(n_base=800, T=16, paired=True, speed_range=sr, seed0=777)
    check_dataset(ev)
    np.savez_compressed("bounce2d_eval.npz", frames=ev["frames"], labels=ev["labels"],
                        meta=ev["meta"], **{f"gt_{k}": v for k, v in ev["gt"].items()})
    print(f"\nSaved bounce2d_eval.npz  frames={ev['frames'].shape}")
