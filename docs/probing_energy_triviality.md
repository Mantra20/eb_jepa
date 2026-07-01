# Triviality test: is energy encoded beyond velocity?

**Experiment:** Exp A (`main_simt12_seed1`, sim_coeff_t=12, checkpoint e-300)
**Question:** does the AC-Video-JEPA latent encode kinetic energy `E = ½‖v‖²`
as a quantity in its own right, or is it merely a mathematical consequence of
the latent already encoding `v`?

Reproduce with:

```bash
python scripts/triviality_test.py --run-dir runs/main_simt12_seed1 --checkpoint e-300.pth.tar
```

## Methodology

Three probes trained on the same latent `X` (time-averaged over T), compared
against each other:

1. **Direct quadratic probe:** `latent → E` via
   `PolynomialFeatures(degree=2) + Ridge`. Captures any quadratic relationship
   between latent coordinates and energy, including energy being a "hidden"
   non-linear combination of the latent.
2. **Indirect probe:** `latent → v̂` (linear), then `Ê = ½‖v̂‖²`. If energy is
   nothing more than a consequence of a linearly encoded v, this method should
   suffice.
3. **Direct linear probe** (baseline): `latent → E`, without quadratic terms.

**Non-triviality criterion:** if the direct probe (1) clearly outperforms the
indirect probe (2), energy is encoded beyond what a linear reconstruction of
velocity can explain — a signal genuinely carried by the latent, not a mere
by-product of v.

## Bugs encountered and fixes

| # | Bug | Cause | Fix |
|---|---|---|---|
| 1 | `KeyError: 'velocity'` | `gt_n` stores fields flat (`vx`, `vy`, `E`); there are no `velocity`/`energy` keys | `v = np.stack([gt_n["vx"], gt_n["vy"]], axis=-1)`, `E = gt_n["E"]` |
| 2 | `ValueError: Found array with dim 3` | `lat_n` has shape `(N, T, D) = (800, 16, 512)` — one latent per frame, not per trajectory | Time average: `X = lat_n.mean(axis=1)` → `(800, 512)`, aligned with `v_mean`/`E_mean` |
| 3 | Inconsistent R² values (`R²_direct=0.569`, `R²_indirect=-1.639`, suspicious delta) | Fixed `alpha=1.0` is badly miscalibrated for a target `E_mean` of absolute scale `~1e-3` (absolute variance `1.96e-6`) | `RidgeCV(alphas=np.logspace(-6, 2, 25))` instead of `Ridge(alpha=1.0)` — after first checking that `CV(E_mean)=0.70` and `CV(v_mean)=0.43` (the signal varies fine *relatively*; only the absolute scale is small) |
| 4 | `v̂ → Ê=½‖v̂‖²` stays bad (`R²=-1.84`) even after recalibration | **Jensen bias**: `v_mean = mean(v_t)` partially cancels the vx/vy components at every bounce (direction change), so `‖v_mean‖² ≠ mean(‖v_t‖²)`. Verified: `R²(E_mean, 0.5·speed_mean²) = 1.0` vs `R²(E_mean, 0.5‖v_mean‖²) = -1.07` | Predict the **scalar norm** `speed_mean = mean(‖v_t‖)` rather than the vector `v_mean`, then `Ê = ½·speed_hat²` |
| 5 | Overfitting risk (131k quadratic features for ~640 training samples) | `PolynomialFeatures(degree=2)` on 512 dims blows up the dimensionality | Permutation test: shuffle `E_mean`, redo the CV. `R²_shuffled = -0.347 ± 0.054`, far from `0.570` → confirms the signal is not an over-parameterization artifact |

## Final results (Exp A, e-300)

| Probe | R² | Interpretation |
|---|---|---|
| Direct quadratic (latent → E) | **0.570** | Real signal — validated by the permutation test |
| Direct linear (latent → E) | −0.254 | E not accessible to a simple linear probe |
| Indirect (latent → speed → Ê) | −0.044 | speed itself not linearly decodable at this aggregation level |
| **Delta (direct − indirect)** | **+0.644** | E encoded beyond a linear reconstruction of speed |
| Negative control (shuffled E, 5 runs) | −0.347 ± 0.054 | Confirms 0.570 is not a dimensionality artifact |

**Conclusion:** kinetic energy is encoded in the latent in a non-linear,
non-trivial way — it is not simply recoverable via a linear reconstruction of
speed followed by computing its norm. This result is consistent with the
independent measurement from the official probe suite
(`energy_R2_quadratic=0.750` vs `energy_R2_linear≈-0.056`), obtained on a
potentially different temporal aggregation of the latent — **a point to verify**
before quoting the two numbers side by side in a final report (a likely
aggregation-method gap between the two protocols, not yet audited).

## Methodological limit to keep in mind

The triviality test as designed answers "does E derive linearly from v?", not
"did the network learn a concept of energy conservation". A non-trivial
quadratic signal is a necessary but not sufficient condition for claiming a
geometric representation of conservation — consistent with the interpretation
adopted elsewhere in the project (emergent but derivable from velocity, not a
geometric conservation signal in the strict sense).
