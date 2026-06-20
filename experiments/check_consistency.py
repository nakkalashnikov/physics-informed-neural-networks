"""
DIAGNOSTIC: are the analytic data labels CONSISTENT with the physics.py residuals?

The hybrid loss trains the net to (a) match analytic labels u and (b) satisfy the PDE +
flux BC. If the ground-truth analytic field does NOT itself satisfy the residuals we enforce,
the two losses pull toward different solutions and training settles at a compromise — exactly
the ~60% data / ~77% VAL plateau that no optimizer trick (clip, lr, sigma) could break.

Two checks on the EXACT analytic field (no network involved):
  1. Interior PDE residual  R = u_t - Fo*u_xx - (Fo/AR^2)*u_yy   (expect ~0: sanity)
  2. Flux BC at yhat=0:      does u_y(0+) == -Q_star*ghat ?       (the decisive one)

If check 2 fails (u_y(0+) ~ 0 instead of -Q*ghat) the cosine-basis labels carry the source as
a boundary delta (insulated, u_y(0)=0) while physics.py asks for a gradient flux — inconsistent.
"""

from __future__ import annotations

import numpy as np
import yaml

from analytic import _ghat, fourier_field_u
from nondim import PiGroups
from trajectory import sample_trajectory


def main() -> None:
    cfg = yaml.safe_load(open("config.yaml"))
    sf = float(cfg["physics"]["sigma_factor"])

    # One representative case (mid-range pi-groups, one fixed trajectory)
    pi = PiGroups(Fo=5e-3, AR=0.15, w=0.05)
    traj = sample_trajectory(cfg, np.random.default_rng(0))
    Fo, AR, Q = pi.Fo, pi.AR, pi.Q_star
    sigma_star = pi.sigma_star(sf)
    print(f"case: Fo={Fo:g} AR={AR} w={pi.w} | Q*={Q:.3f} sigma*={sigma_star:.4f}\n")

    # ── CHECK 1: interior PDE residual ───────────────────────────────────────
    nx = ny = 81
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0.2, 0.8, 31)
    u = fourier_field_u(x, y, t, traj, pi, cfg)          # (t, y, x)
    dt, dy, dx = t[1] - t[0], y[1] - y[0], x[1] - x[0]

    u_t = np.gradient(u, dt, axis=0)
    u_x = np.gradient(u, dx, axis=2)
    u_xx = np.gradient(u_x, dx, axis=2)
    u_y = np.gradient(u, dy, axis=1)
    u_yy = np.gradient(u_y, dy, axis=1)
    R = u_t - Fo * u_xx - (Fo / AR**2) * u_yy

    # interior mask (away from edges where FD is one-sided / source layer sits)
    iy = (y >= 0.15) & (y <= 0.85)
    ix = (x >= 0.10) & (x <= 0.90)
    Rin = R[:, iy][:, :, ix]
    utin = u_t[:, iy][:, :, ix]
    rms = lambda a: float(np.sqrt(np.mean(a**2)))
    print("── CHECK 1: interior PDE residual ──")
    print(f"  RMS(R) / RMS(u_t) = {rms(Rin)/rms(utin):.2%}   (expect ~0; >~10% = interior bug)\n")

    # ── CHECK 2: flux BC at yhat=0 (the decisive one) ────────────────────────
    t0 = 0.5
    xf = np.linspace(0, 1, 201)
    yf = np.linspace(0, 0.1, 41)                          # fine near boundary, dy=0.0025
    uf = fourier_field_u(xf, yf, np.array([t0]), traj, pi, cfg)[0]   # (y, x)
    uy0 = (uf[1, :] - uf[0, :]) / (yf[1] - yf[0])         # one-sided d u/d yhat at yhat=0

    xb0 = float(traj(np.array([t0]))[0])
    ghat0 = _ghat(xf - xb0, sigma_star)
    expected = -Q * ghat0                                 # what physics.py BC demands

    jpk = int(np.argmax(ghat0))                           # x at the source peak
    print("── CHECK 2: flux BC  u_y(0) vs -Q*ghat ──")
    print(f"  at source peak x={xf[jpk]:.3f}:")
    print(f"     analytic u_y(0+)   = {uy0[jpk]:10.3f}")
    print(f"     physics demands    = {expected[jpk]:10.3f}   (-Q*ghat)")
    print(f"     ratio actual/demanded = {uy0[jpk]/expected[jpk]:.3f}  "
          f"(1=consistent, ~0=labels carry no flux)")
    print(f"  RMS over x:  u_y(0+)={rms(uy0):.3f}   -Q*ghat={rms(expected):.3f}   "
          f"ratio={rms(uy0)/rms(expected):.3f}")
    bc_res = uy0 + Q * ghat0                              # physics.py residual on TRUE field
    print(f"  RMS BC residual on TRUE field = {rms(bc_res):.3f}  "
          f"(should be ~0 if labels satisfy the BC)\n")

    # also: how u_y evolves into the domain — does it ever reach -Q*ghat?
    print("  u_y at increasing yhat (peak x), vs demanded -Q*ghat at y=0:")
    for j in range(0, 12, 2):
        uy = (uf[j + 1, jpk] - uf[j, jpk]) / (yf[1] - yf[0])
        print(f"     yhat={yf[j]:.4f}:  u_y={uy:9.3f}")
    print(f"     (-Q*ghat at peak = {expected[jpk]:.3f})")


if __name__ == "__main__":
    main()
