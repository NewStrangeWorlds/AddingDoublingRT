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


def compute_doubling_count(tau, omega, mu_min):
    """Number of doublings: omega-adaptive rule plus the extinction floor
    tau0 = tau / 2**nn <= mu_min / 2 (mirrors adrt::computeDoublingCount)."""
    nn = int(np.log(tau) / np.log(2.0)) + _compute_ipow0(omega)
    n_ext = int(np.ceil(np.log2(tau / mu_min))) + 1
    return max(1, nn, n_ext)


def _phi(u):
    """expm1(u) / u with phi(0) = 1."""
    small = jnp.abs(u) < 1e-6
    safe = jnp.where(small, 1.0, u)
    return jnp.where(small, 1.0 + 0.5 * u, jnp.expm1(safe) / safe)


def _path_integral(tau0, d, e_i, e_j):
    """int_0^tau0 exp(-(tau0-t)/mu_i) exp(-t/mu_j) dt = (e_j - e_i) / d with
    d = 1/mu_i - 1/mu_j: tau0 e_i phi(tau0 d) for small |tau0 d| (no
    cancellation), difference form otherwise (no overflow of expm1)."""
    u = tau0 * d
    small = jnp.abs(u) < 1.0
    u_safe = jnp.where(small, u, 0.0)
    d_safe = jnp.where(small, 1.0, d)
    return jnp.where(small, tau0 * e_i * _phi(u_safe), (e_j - e_i) / d_safe)


def doubling_start(tau0, omega, Spp, Spm, mu,
                   F_top=None, p_plus=None, p_minus=None, solar_mu=None):
    """Thin-layer initialisation of the doubling recursion (port of
    adrt::initDoublingStart).

    Exact extinction e_i = exp(-tau0/mu_i), exact single scattering, Taylor
    double scattering tau0^2/2 * S^2, and a thermal source consistent with the
    absorbed fraction of the operators (Kirchhoff) through O(tau0^2):
        y_i = (1-omega) [ (1-e_i) + tau0^2/2 sum_k (Spp+Spm)_ik / mu_k ].
    Extinction and single scattering are exact for any tau0/mu, so the start
    never leaves the physical range (the former first-order start
    T = I - tau0*Gpp blew up under doubling for tau0 > mu_min).

    Broadcasts over leading batch dimensions:
        tau0, omega: (...), Spp, Spm: (..., N, N), mu: (N,),
        F_top: (...), p_plus, p_minus: (N,).
    Returns (R, T, y, z, s_up_sol, s_down_sol).
    """
    tau0 = jnp.asarray(tau0)
    omega = jnp.asarray(omega)
    inv_mu = 1.0 / mu
    N = mu.shape[0]

    t0v = tau0[..., None]                      # (..., 1)
    x = t0v * inv_mu                           # (..., N)
    e = jnp.exp(-x)
    a = -jnp.expm1(-x)                         # 1 - e

    d = inv_mu[:, None] - inv_mu[None, :]      # 1/mu_i - 1/mu_j
    sm = inv_mu[:, None] + inv_mu[None, :]     # 1/mu_i + 1/mu_j
    t0m = tau0[..., None, None]
    int_T = _path_integral(t0m, d, e[..., :, None], e[..., None, :])
    int_R = t0m * _phi(-t0m * sm)

    T = jnp.eye(N, dtype=e.dtype) * e[..., :, None] + Spp * int_T
    R = Spm * int_R

    h = 0.5 * tau0 * tau0
    hm = h[..., None, None]
    T = T + hm * (Spp @ Spp + Spm @ Spm)
    R = R + hm * (Spp @ Spm + Spm @ Spp)

    omv = omega[..., None]
    hv = h[..., None]
    d2 = jnp.sum((Spp + Spm) * inv_mu, axis=-1)          # sum_k S_ik / mu_k
    y = (1.0 - omv) * (a + hv * d2)

    # z_i = (1-omega) [ mu_i (1-e_i) - tau0/2 (1+e_i) ] = -(1-omega) mu_i x^3/12 + ...
    x_small = 1e-3 if e.dtype == jnp.float64 else 3e-2
    slope_series = mu * x ** 3 * (-1.0 / 12.0 + x * (1.0 / 24.0 - x / 80.0))
    slope_direct = mu * a - 0.5 * t0v * (1.0 + e)
    z = (1.0 - omv) * jnp.where(x < x_small, slope_series, slope_direct)

    if F_top is None:
        zeros = jnp.zeros_like(y)
        return R, T, y, z, zeros, zeros

    F_top = jnp.asarray(F_top)
    inv_mu0 = 1.0 / solar_mu
    e0 = jnp.exp(-t0v * inv_mu0)
    int_up = t0v * _phi(-t0v * (inv_mu0 + inv_mu))
    int_dn = _path_integral(t0v, inv_mu - inv_mu0, e, e0)

    src_up = omv * F_top[..., None] * p_minus * inv_mu
    src_dn = omv * F_top[..., None] * p_plus * inv_mu

    d_up = jnp.einsum("...ik,...k->...i", Spp, src_up) + jnp.einsum("...ik,...k->...i", Spm, src_dn)
    d_dn = jnp.einsum("...ik,...k->...i", Spp, src_dn) + jnp.einsum("...ik,...k->...i", Spm, src_up)

    s_up_sol = src_up * int_up + hv * d_up
    s_down_sol = src_dn * int_dn + hv * d_dn
    return R, T, y, z, s_up_sol, s_down_sol


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
    Spp = con * (Ppp @ C) / mu[:, None]
    Spm = con * (Ppm @ C) / mu[:, None]

    nn = compute_doubling_count(tau, omega, float(jnp.min(mu)))
    xfac = 1.0 / (2.0 ** nn)
    tau0 = tau * xfac

    has_solar = (solar_flux > 0.0 and solar_mu > 0.0
                 and p_plus_solar is not None and p_minus_solar is not None)
    F_top = solar_flux * jnp.exp(-tau_cumulative / solar_mu) if has_solar else 0.0

    if has_solar:
        R_k, T_k, y_k, z_k, s_up_sol_k, s_down_sol_k = doubling_start(
            tau0, omega, Spp, Spm, mu, F_top, p_plus_solar, p_minus_solar, solar_mu)
    else:
        R_k, T_k, y_k, z_k, s_up_sol_k, s_down_sol_k = doubling_start(
            tau0, omega, Spp, Spm, mu)

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
