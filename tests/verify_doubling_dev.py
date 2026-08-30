"""Verification suite for the reworked doubling initialisation (AddingDoublingRT dev).

Run from the repository root:  python verify_doubling_dev.py

Exercises the shipped src_jax implementation (doubling_start / compute_doubling_count),
which is a line-for-line port of adrt::initDoublingStart / adrt::computeDoublingCount.

Checks
  1. omega -> 0 continuity: the scattering branch must reproduce the analytic
     omega <= 0 branch. This is the test that failed before 6cfbaaf.
  2. Accuracy at the count the rule selects, over a (tau, omega) grid.
  3. Flux conservation of the converged operators for a conservative layer,
     including anisotropic phase functions.
  4. An omega-scaled extinction floor, which removes the residual ~1e-4 shoulder
     at omega just below 0.01 and the accuracy jump across that branch boundary.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src_jax.doubling import doubling_start, compute_doubling_count
from src_jax.quadrature import gauss_legendre, precompute_legendre_polynomials
from src_jax.phase_matrix import compute_phase_matrices

NMOM = 16


def quadrature(N):
    mu, wt = gauss_legendre(N)
    return np.asarray(mu, float), np.asarray(wt, float)


def phase_ops(N, omega, g_hg=0.0):
    """Spp, Spm as adrt::doubling builds them: 2*pi*omega*P*w_j/mu_i."""
    mu, wt = quadrature(N)
    Pl = precompute_legendre_polynomials(NMOM, jnp.asarray(mu))
    chi = (jnp.asarray(g_hg ** np.arange(NMOM)) if g_hg
           else jnp.zeros(NMOM).at[0].set(1.0))
    Ppp, Ppm = compute_phase_matrices(chi, jnp.asarray(mu), jnp.asarray(wt), Pl)
    f = 2.0 * np.pi * omega * wt[None, :] / mu[:, None]
    return np.asarray(Ppp, float) * f, np.asarray(Ppm, float) * f


def layer(tau, omega, nn, N=8, g_hg=0.0):
    """Shipped thin-layer start followed by nn doublings."""
    mu, _ = quadrature(N)
    Spp, Spm = phase_ops(N, omega, g_hg)
    tau0 = tau / 2.0 ** nn
    R, T, y, z, _, _ = doubling_start(
        jnp.asarray(tau0), jnp.asarray(omega),
        jnp.asarray(Spp), jnp.asarray(Spm), jnp.asarray(mu))
    R, T, y, z = (np.asarray(a, float) for a in (R, T, y, z))
    I, g = np.eye(N), 0.5 * tau0
    for _ in range(nn):
        TG = np.linalg.solve((I - R @ R).T, T.T).T
        TGR = TG @ R
        zpgy = z + g * y
        R, T, y, z, g = (R + TGR @ T, TG @ T, TG @ y + TGR @ y + y,
                         (TG @ zpgy - TGR @ zpgy) + z - g * y, 2.0 * g)
    return R, T, y


def n_omega_floor(tau, omega, mu_min, c=7.6):
    """computeDoublingCount with the extinction floor scaled by sqrt(omega).

    The accumulated error of the O(tau0^3) start behaves as omega*(tau0/mu_min)^2,
    so the floor that delivers a uniform tolerance is
        tau0/mu_min <= 2^-c / sqrt(omega),
    rather than the omega-independent tau0 <= mu_min/2. c = 7.6 targets ~1 ppm.
    """
    ipow0 = 4 if omega < 0.01 else (10 if omega < 0.1 else 16)
    n_ms = int(np.log(tau) / np.log(2.0)) + ipow0
    n_ext = int(np.ceil(np.log2(tau / mu_min) + 0.5 * np.log2(max(omega, 1e-8)) + c))
    return max(1, n_ms, n_ext)


def check_omega_continuity():
    print("1. omega -> 0 continuity (omega = 1e-8; residual should be O(omega))")
    print("   N   tau     nn    max|delta| vs analytic   verdict")
    ok = True
    for N in (8, 16):
        mu, _ = quadrature(N)
        for tau in (0.1, 1.0, 5.0, 20.0):
            nn = compute_doubling_count(tau, 1e-8, float(mu.min()))
            R, T, y = layer(tau, 1e-8, nn, N)
            trans = np.exp(-tau / mu)
            err = max(np.abs(T - np.diag(trans)).max(),
                      np.abs(y - (1.0 - trans)).max(), np.abs(R).max())
            ok &= err < 1e-6
            print(f"  {N:2d} {tau:6.1f}   {nn:3d}      {err:.3e}          "
                  f"{'PASS' if err < 1e-6 else 'FAIL'}")
    return ok


def check_accuracy():
    mu, _ = quadrature(8)
    mu_min = float(mu.min())
    grid = [(t, o) for t in (0.05, 0.5, 2.0, 5.0, 20.0, 100.0)
            for o in (1e-6, 1e-4, 3e-3, 9e-3, 1.1e-2, 0.05, 0.09, 0.11, 0.5, 0.9, 0.999)]
    stats = {}
    for tag, rule in (("shipped", compute_doubling_count),
                      ("omega-scaled floor", n_omega_floor)):
        errs, counts = [], []
        for tau, om in grid:
            Rr, Tr, yr = layer(tau, om, 36)
            s = max(np.abs(Rr).max(), np.abs(Tr).max(), np.abs(yr).max())
            nn = rule(tau, om, mu_min)
            R, T, y = layer(tau, om, nn)
            errs.append(max(np.abs(R - Rr).max(), np.abs(T - Tr).max(),
                            np.abs(y - yr).max()) / s)
            counts.append(nn)
        stats[tag] = (np.array(errs), np.array(counts))
    print(f"\n2. accuracy at the selected count, {len(grid)} (tau, omega) points, N=8")
    print("   rule                  max err     median err   total doublings")
    for tag, (e, n) in stats.items():
        print(f"   {tag:<20s}  {e.max():.2e}    {np.median(e):.2e}     {n.sum():5d}")
    i = stats["shipped"][0].argmax()
    print(f"   shipped worst case: tau={grid[i][0]:g}, omega={grid[i][1]:g}, "
          f"nn={stats['shipped'][1][i]}, err={stats['shipped'][0][i]:.2e}")
    print("\n   across the omega = 0.01 branch boundary at tau = 20:")
    for om in (0.009, 0.011):
        e = []
        for rule in (compute_doubling_count, n_omega_floor):
            nn = rule(20.0, om, mu_min)
            Rr, Tr, yr = layer(20.0, om, 36)
            s = max(np.abs(Rr).max(), np.abs(Tr).max(), np.abs(yr).max())
            R, T, y = layer(20.0, om, nn)
            e.append((nn, max(np.abs(R - Rr).max(), np.abs(T - Tr).max(),
                              np.abs(y - yr).max()) / s))
        print(f"     omega={om:<6g} shipped nn={e[0][0]:2d} err={e[0][1]:.1e}   |   "
              f"omega-scaled nn={e[1][0]:2d} err={e[1][1]:.1e}")
    return stats


def check_conservation():
    print("\n3. flux conservation at omega = 1:  sum_i w_i mu_i (T+R)_ij = w_j mu_j")
    print("   tau     g_HG    nn    max_j relative violation")
    mu, wt = quadrature(8)
    w = wt * mu
    ok = True
    for tau, g_hg in ((0.5, 0.0), (5.0, 0.0), (5.0, 0.7), (20.0, 0.7), (1.0, 0.9)):
        nn = compute_doubling_count(tau, 1.0, float(mu.min()))
        R, T, _ = layer(tau, 1.0, nn, 8, g_hg)
        viol = np.abs((w[:, None] * (T + R)).sum(axis=0) / w - 1.0).max()
        ok &= viol < 1e-5
        print(f"  {tau:6.1f}   {g_hg:4.1f}   {nn:3d}     {viol:.3e}")
    return ok


if __name__ == "__main__":
    a = check_omega_continuity()
    check_accuracy()
    b = check_conservation()
    print(f"\ncontinuity: {'PASS' if a else 'FAIL'}   conservation: {'PASS' if b else 'FAIL'}")
