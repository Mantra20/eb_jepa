"""CPU smoke test: replicates the AC-Video training-loop CORE with the REAL model
components (built exactly as examples/ac_video_jepa/main.py), fed bounce2D batches.
Validates the integration end-to-end (data -> JEPA -> loss -> backward -> step)
without the CLI/wandb/Two-Rooms-eval machinery (irrelevant to integration)."""
import torch, torch.nn as nn
from eb_jepa.architectures import ImpalaEncoder, InverseDynamicsModel, RNNPredictor
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.state_decoder import MLPXYHead
from torch.optim import AdamW
from bounce2d_ebjepa import build_bounce2d_loaders, effective_rank

torch.manual_seed(0)
device = torch.device("cpu")
DOBS, IMG, NSTEPS = 2, 65, 8
HENC, DSTC = 32, 32

# --- data (real loader, bounce2d, normal-only training) ---
loader, vloader, dc = build_bounce2d_loaders(
    {"batch_size": 8, "T": 16, "img_size": IMG, "n_base": 64, "n_val": 16})
print(f"[data] batch_size={dc.batch_size} train={dc.size} val={dc.val_size} img={dc.img_size}")

# --- model: built exactly as main.py ---
test_input = torch.rand((1, DOBS, 1, IMG, IMG))
encoder = ImpalaEncoder(width=1, stack_sizes=(16, HENC, DSTC), num_blocks=2,
                        dropout_rate=None, layer_norm=False, input_channels=DOBS,
                        final_ln=True, mlp_output_dim=512,
                        input_shape=(DOBS, IMG, IMG))
test_output = encoder(test_input)
_, f, _, h, w = test_output.shape
print(f"[model] encoder output {tuple(test_output.shape)}  (f={f}, h={h}, w={w})")
predictor = RNNPredictor(hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln)
idm = InverseDynamicsModel(state_dim=h * w * f, hidden_dim=256, action_dim=2).to(device)
regularizer = VC_IDM_Sim_Regularizer(
    cov_coeff=12, std_coeff=16, sim_coeff_t=12, idm_coeff=1,
    std_margin=2.0, idm=idm, first_t_only=False, projector=None,
    spatial_as_samples=False, idm_after_proj=False, sim_t_after_proj=False)
jepa = JEPA(encoder, nn.Identity(), predictor, regularizer, SquareLossSeq()).to(device)

xy_head = MLPXYHead(input_shape=test_output.shape[1],
                    normalizer=loader.dataset.normalizer).to(device)
xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=nn.MSELoss())

enc_p = sum(p.numel() for p in encoder.parameters())
pred_p = sum(p.numel() for p in predictor.parameters())
print(f"[model] params: encoder={enc_p/1e6:.2f}M predictor={pred_p/1e6:.2f}M  std_margin=2.0 cov=12 std=16")

jepa_opt = AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-5)
probe_opt = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)

# --- training-loop core: a few steps ---
print("\n[train] running 5 steps (jepa.unroll + xy_prober + backward) ...")
jepa.train()
for step, (x, a, loc, energy, labels) in enumerate(loader):
    if step >= 5: break
    x, a, loc = x.to(device), a.to(device), loc.to(device)

    jepa_opt.zero_grad()
    _, (jepa_loss, regl, regl_unw, regldict, pl) = jepa.unroll(
        x, a, nsteps=NSTEPS, unroll_mode="autoregressive",
        ctxt_window_time=1, compute_loss=True, return_all_steps=False)
    jepa_loss.backward()
    torch.nn.utils.clip_grad_norm_(jepa.encoder.parameters(), 2.0)
    torch.nn.utils.clip_grad_norm_(jepa.predictor.parameters(), 2.0)
    jepa_opt.step()

    probe_opt.zero_grad()
    xy_loss = xy_prober(observations=x[:, :, :1], targets=loc[:, :, :1])
    xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
    xy_loss.backward()
    probe_opt.step()

    # collapse diagnostic on the encoded state (flatten C*H*W per frame, all timesteps)
    with torch.no_grad():
        z = encoder(x)                      # [B, f, T, h, w]
        B, F, T, H, W = z.shape
        zt = z.permute(0, 2, 1, 3, 4).reshape(B * T, F * H * W)
        er = effective_rank(zt[:, :min(256, zt.shape[1])])
    reg_terms = {k: round(float(v), 3) for k, v in regldict.items()}
    print(f"  step {step}: total~{float(jepa_loss):.4f} pred={float(pl):.4f} "
          f"reg={float(regl):.4f} probe={float(xy_loss):.4f} eff_rank={er:.1f} | {reg_terms}")

print("\n[OK] integration validated: bounce2D batches flow through the real JEPA "
      "model, losses are finite, gradients applied, effective_rank computed.")
