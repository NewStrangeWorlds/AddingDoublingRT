"""Doubling algorithm: compute R, T, source vectors for a single
homogeneous layer via iterative doubling."""

import jax.numpy as jnp
import numpy as np

PI = jnp.pi


def _compute_ipow0(omega):
    """Adaptive number of initial doublings based on single-scattering albedo."""
    if omega < 0.01:
        return 4
    if omega < 0.1:
        return 10
    return 16


def doubling(tau, omega, B_top, B_bottom, Ppp, Ppm, mu, weights,
             solar_flux=0.0, solar_mu=0.0, tau_cumulative=0.0,
             p_plus_solar=None, p_minus_solar=None):
    """Compute layer R, T matrices and source vectors via iterative doubling.

    Args:
        tau: Optical depth of the layer.
        omega: Single-scattering albedo.
        B_top: Planck function at layer top.
        B_bottom: Planck function at layer bottom.
        Ppp: Phase matrix P++, shape (N, N).
        Ppm: Phase matrix P+-, shape (N, N).
        mu: Quadrature nodes, shape (N,).
        weights: Quadrature weights, shape (N,).
        solar_flux: Solar flux at TOA.
        solar_mu: cos(solar zenith angle).
        tau_cumulative: Cumulative optical depth above this layer.
        p_plus_solar: Solar phase vector p+, shape (N,).
        p_minus_solar: Solar phase vector p-, shape (N,).

    Returns:
        dict with keys: R_ab, R_ba, T_ab, T_ba, s_up, s_down,
        s_up_solar, s_down_solar, is_scattering.
    """
    N = mu.shape[0]
    I_mat = jnp.eye(N)

    B_bar = (B_bottom + B_top) / 2.0
    B_d = (B_bottom - B_top) / tau if tau > 0.0 else 0.0

    # Zero optical depth
    if tau <= 0.0:
        return {
            "R_ab": jnp.zeros((N, N)),
            "R_ba": jnp.zeros((N, N)),
            "T_ab": I_mat,
            "T_ba": I_mat,
            "s_up": jnp.zeros(N),
            "s_down": jnp.zeros(N),
            "s_up_solar": jnp.zeros(N),
            "s_down_solar": jnp.zeros(N),
            "is_scattering": False,
        }

    # Pure absorption
    if omega <= 0.0:
        tex = -tau / mu
        trans = jnp.where(tex > -200.0, jnp.exp(tex), 0.0)
        T_diag = jnp.diag(trans)
        one_minus_t = 1.0 - trans
        slope_term = mu * one_minus_t - 0.5 * tau * (1.0 + trans)
        s_up = B_bar * one_minus_t + B_d * slope_term
        s_down = B_bar * one_minus_t - B_d * slope_term

        return {
            "R_ab": jnp.zeros((N, N)),
            "R_ba": jnp.zeros((N, N)),
            "T_ab": T_diag,
            "T_ba": T_diag,
            "s_up": s_up,
            "s_down": s_down,
            "s_up_solar": jnp.zeros(N),
            "s_down_solar": jnp.zeros(N),
            "is_scattering": False,
        }

    # Scattering layer
    omega = float(np.clip(omega, 0.0, 1.0))
    con = 2.0 * omega * PI

    C = jnp.diag(weights)
    PppC = Ppp @ C
    temp = I_mat - con * PppC
    Gpp = temp / mu[:, None]

    PpmC = Ppm @ C
    Gpm = con * PpmC / mu[:, None]

    nn = int(np.log(tau) / np.log(2.0)) + _compute_ipow0(omega)
    if nn < 1:
        nn = 1
    xfac = 1.0 / (2.0 ** nn)
    tau0 = tau * xfac

    has_solar = (solar_flux > 0.0 and solar_mu > 0.0
                 and p_plus_solar is not None and p_minus_solar is not None)
    F_top = solar_flux * jnp.exp(-tau_cumulative / solar_mu) if has_solar else 0.0

    R_k = tau0 * Gpm
    T_k = I_mat - tau0 * Gpp

    y_k = (1.0 - omega) * tau0 / mu
    z_k = jnp.zeros(N)

    if has_solar:
        base = omega * tau0 / mu * F_top
        s_up_sol_k = base * p_minus_solar
        s_down_sol_k = base * p_plus_solar
    else:
        s_up_sol_k = jnp.zeros(N)
        s_down_sol_k = jnp.zeros(N)

    g_k = 0.5 * tau0
    gamma_sol = jnp.exp(-tau0 / solar_mu) if has_solar else 0.0

    for _ in range(nn):
        R_sq = R_k @ R_k
        I_minus_R2 = I_mat - R_sq

        # TG = T_k @ (I - R^2)^{-1}  i.e. solve (I - R^2)^T X^T = T_k^T
        TG = jnp.linalg.solve(I_minus_R2.T, T_k.T).T
        TGR = TG @ R_k

        R_new = R_k + TGR @ T_k
        T_new = TG @ T_k

        zpgy = z_k + g_k * y_k
        TG_zpgy = TG @ zpgy
        TGR_zpgy = TGR @ zpgy
        TG_y = TG @ y_k
        TGR_y = TGR @ y_k

        z_new = (TG_zpgy - TGR_zpgy) + z_k - g_k * y_k
        y_new = TG_y + TGR_y + y_k

        if has_solar:
            R_sdown = R_k @ s_down_sol_k
            R_sup = R_k @ s_up_sol_k

            rhs_up = R_sdown + gamma_sol * s_up_sol_k
            rhs_down = gamma_sol * R_sup + s_down_sol_k

            s_up_sol_new = TG @ rhs_up + s_up_sol_k
            s_down_sol_new = TG @ rhs_down + gamma_sol * s_down_sol_k
            gamma_sol = gamma_sol * gamma_sol
        else:
            s_up_sol_new = jnp.zeros(N)
            s_down_sol_new = jnp.zeros(N)

        R_k = R_new
        T_k = T_new
        y_k = y_new
        z_k = z_new
        s_up_sol_k = s_up_sol_new
        s_down_sol_k = s_down_sol_new
        g_k = 2.0 * g_k

    return {
        "R_ab": R_k,
        "R_ba": R_k,
        "T_ab": T_k,
        "T_ba": T_k,
        "s_up": y_k * B_bar + z_k * B_d,
        "s_down": y_k * B_bar - z_k * B_d,
        "s_up_solar": s_up_sol_k,
        "s_down_solar": s_down_sol_k,
        "is_scattering": True,
    }
