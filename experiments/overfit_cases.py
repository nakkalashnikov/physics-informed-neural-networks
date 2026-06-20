"""
DIAGNOSTIC 2: how many cases can the OPERATOR memorise?  (branch / generalisation test)

overfit_one.py proved the trunk can represent a single field (~3% with sigma 3.5).
So the failure is not single-field representation — it is learning the mapping
   branch:  (trajectory samples ++ pi-groups)  ->  latent that selects the right field
across many cases. This script fits N distinct cases at once (pure supervised, no physics)
and reports the train rel-L2. Sweep N and find where it breaks:

    python overfit_cases.py --n-cases 1
    python overfit_cases.py --n-cases 8
    python overfit_cases.py --n-cases 64
    python overfit_cases.py --n-cases 512

  * rel-L2 stays low as N grows  -> branch capacity is fine; full-training failure is
                                    optimisation / loss-balance / not-enough-steps.
  * rel-L2 climbs toward ~70% at modest N -> the BRANCH is the bottleneck: it cannot encode
                                    the operator. Fix = wider/deeper branch or larger latent p.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import yaml

from model import build_model, BranchNet, TrunkNet
from nondim import PiGroups, normalize, sample_pi_groups
from trajectory import sample_trajectory, sample_at_nodes
from analytic import fourier_field_u


class NonlinearDeepONet(nn.Module):
    """
    Diagnostic-only variant: same branch + trunk, but the LINEAR dot-product decoder
    u = (b . tau) * t*  is replaced by a NONLINEAR merge head
        u = MLP([b, tau]) * t*
    so each case gets an adapted (non-linear-combination) basis rather than fixed
    coefficients on a shared linear basis. If this breaks the n-width wall, it proves
    the fix is the decoder structure, not size.
    """

    def __init__(self, cfg: dict, head_width: int = 256):
        super().__init__()
        k = int(cfg["trajectory"]["k_sensors"])
        p = int(cfg["branch"]["out_dim"])
        self.branch = BranchNet(in_dim=k + 3, hidden=list(cfg["branch"]["hidden"]), out_dim=p)
        self.trunk = TrunkNet(
            rff_m=int(cfg["trunk"]["rff_num_features"]),
            sigma_start=float(cfg["trunk"]["rff_sigma_start"]),
            n_blocks=int(cfg["trunk"]["n_pirate_blocks"]),
            width=int(cfg["trunk"]["width"]),
            out_dim=p,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * p, head_width), nn.Tanh(),
            nn.Linear(head_width, head_width), nn.Tanh(),
            nn.Linear(head_width, 1),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_sigma(self, sigma: float) -> None:
        self.trunk.set_sigma(sigma)

    def forward(self, branch_in: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch_in)
        tau = self.trunk(coords)
        out = self.head(torch.cat([b, tau], dim=-1))
        return coords[:, 2:3] * out          # hard IC (u=0 at t*=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--sigma-end", type=float, default=3.5)
    ap.add_argument("--batch", type=int, default=8192, help="points per gradient step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (matches trainer; 0=off)")
    ap.add_argument("--p", type=int, default=None, help="override latent width p (branch+trunk out_dim)")
    ap.add_argument("--branch-hidden", default=None, help="override branch MLP, e.g. 512,512,512,512")
    ap.add_argument("--nonlinear", action="store_true", help="use nonlinear merge head instead of dot product")
    ap.add_argument("--source-frame", action="store_true",
                    help="feed trunk x-x_b(t) instead of x (make trunk source-aware)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = yaml.safe_load(open("config.yaml"))
    if args.p is not None:                       # test the latent-rank (Kolmogorov n-width) hypothesis
        cfg["branch"]["out_dim"] = args.p
        cfg["trunk"]["out_dim"] = args.p
    if args.branch_hidden is not None:           # test whether the BRANCH MLP is the encoder bottleneck
        cfg["branch"]["hidden"] = [int(s) for s in args.branch_hidden.split(",")]
    rng = np.random.default_rng(0)

    k = int(cfg["trajectory"]["k_sensors"])
    nx, ny = 40, 30
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny); t = np.array([0.25, 0.5, 0.75, 1.0])

    # shared grid coords (same for every case), column order [x*, y_hat, t*]
    Tg, Yg, Xg = np.meshgrid(t, y, x, indexing="ij")          # each (nt, ny, nx)
    coords = np.stack([Xg.ravel(), Yg.ravel(), Tg.ravel()], axis=1).astype(np.float32)
    npts = coords.shape[0]

    # ── Build N cases: per-case branch vector + per-case target field ────────
    branch_all = np.empty((args.n_cases, k + 3), dtype=np.float32)
    target_all = np.empty((args.n_cases, npts), dtype=np.float32)
    xb_all = np.zeros((args.n_cases, npts), dtype=np.float32)   # source pos x_b(t) per point
    for c in range(args.n_cases):
        pi = sample_pi_groups(cfg, rng)
        traj = sample_trajectory(cfg, rng)
        branch_all[c] = np.concatenate([sample_at_nodes(traj, k), np.array(normalize(pi, cfg))])
        target_all[c] = fourier_field_u(x, y, t, traj, pi, cfg).reshape(-1)   # (nt,ny,nx)->flat
        if args.source_frame:
            xb_all[c] = traj(coords[:, 2]).astype(np.float32)   # x_b at each point's t*

    coords_t = torch.as_tensor(coords, device=device)
    branch_t = torch.as_tensor(branch_all, device=device)
    target_t = torch.as_tensor(target_all, device=device)
    xb_t = torch.as_tensor(xb_all, device=device)
    tgt_norm = float(torch.linalg.norm(target_t))

    def to_trunk_coords(co: torch.Tensor, case_idx: torch.Tensor, pt_idx: torch.Tensor) -> torch.Tensor:
        """If source-frame: replace x with (x - x_b(t)) so the trunk is source-aware."""
        if not args.source_frame:
            return co
        co = co.clone()
        co[:, 0] = co[:, 0] - xb_t[case_idx, pt_idx]
        return co

    model = (NonlinearDeepONet(cfg) if args.nonlinear else build_model(cfg)).to(device)
    decoder = "NONLINEAR merge head" if args.nonlinear else "linear dot-product"
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sigma_start = float(cfg["trunk"]["rff_sigma_start"])
    print(f"decoder: {decoder}")

    print(f"Overfit {args.n_cases} cases | {n_params:,} params | {npts} pts/case "
          f"| sigma {sigma_start}->{args.sigma_end} | batch {args.batch} | {args.steps} steps\n")

    all_pts = torch.arange(npts, device=device)

    def eval_rel() -> float:
        with torch.no_grad():
            # full forward in chunks to keep memory bounded
            errs_sq, tot_sq = 0.0, 0.0
            for c in range(args.n_cases):
                br = branch_t[c].unsqueeze(0).expand(npts, -1)
                ci_full = torch.full((npts,), c, device=device)
                co = to_trunk_coords(coords_t, ci_full, all_pts)
                pred = model(br, co).squeeze(1)
                errs_sq += float(((pred - target_t[c]) ** 2).sum())
                tot_sq += float((target_t[c] ** 2).sum())
            return (errs_sq ** 0.5) / (tot_sq ** 0.5 + 1e-12)

    for step in range(args.steps):
        frac = step / max(args.steps - 1, 1)
        model.set_sigma(sigma_start + (args.sigma_end - sigma_start) * frac)

        # minibatch: random (case, point) pairs
        ci = torch.randint(args.n_cases, (args.batch,), device=device)
        pidx = torch.randint(npts, (args.batch,), device=device)
        br_mb = branch_t[ci]                         # (B, k+3)
        co_mb = to_trunk_coords(coords_t[pidx], ci, pidx)   # (B, 3) — source-frame if enabled
        tg_mb = target_t[ci, pidx].unsqueeze(1)      # (B, 1)

        pred = model(br_mb, co_mb)
        loss = ((pred - tg_mb) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step % 400 == 0 or step == args.steps - 1:
            print(f"step {step:5d} | train rel-L2 {eval_rel():6.1%} | sigma {model.trunk.fourier.sigma.item():.2f}")

    print(f"\nFINAL train rel-L2 = {eval_rel():.1%}  (over {args.n_cases} cases)")
    print("→ low & flat as N grows: branch OK, look at optimisation/loss-balance")
    print("→ climbs with N        : BRANCH is the bottleneck (capacity)")


if __name__ == "__main__":
    main()
