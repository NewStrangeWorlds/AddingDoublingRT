"""Batched solver: fully JIT-compiled, vectorized across wavenumbers.

Mirrors the CUDA batch solver interface. All wavenumbers share the same
phase function moments but have different optical depths, SSA, and Planck values.
The entire solve (all layers, doubling, adding) compiles into a single XLA program.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .quadrature import gauss_legendre, precompute_legendre_polynomials
from .phase_matrix import compute_phase_matrices, compute_solar_phase_vectors

PI = jnp.pi


def _batch_solve_vec(A, b):
    """Batched solve A x = b where A is (..., N, N) and b is (..., N)."""
    return jnp.linalg.solve(A, b[..., None]).squeeze(-1)


def _right_solve_batched(A, B):
    """Batched right solve: X A = B, returns X = B A^{-1}."""
    return jnp.linalg.solve(
        jnp.swapaxes(A, -2, -1), jnp.swapaxes(B, -2, -1)
    ).swapaxes(-2, -1)


# ============================================================================
#  Core operations (all JIT-compatible, no Python control flow)
# ============================================================================

def _doubling_step(carry, _):
    """Single doubling iteration. Used inside jax.lax.scan."""
    R_k, T_k, y_k, z_k, g_k, s_up_sol, s_down_sol, gamma_sol = carry
    N = R_k.shape[-1]
    I_mat = jnp.eye(N)

    R_sq = jnp.einsum("bij,bjk->bik", R_k, R_k)
    I_minus_R2 = I_mat[None] - R_sq

    TG = _right_solve_batched(I_minus_R2, T_k)
    TGR = jnp.einsum("bij,bjk->bik", TG, R_k)

    R_new = R_k + jnp.einsum("bij,bjk->bik", TGR, T_k)
    T_new = jnp.einsum("bij,bjk->bik", TG, T_k)

    zpgy = z_k + g_k[:, None] * y_k
    TG_zpgy = jnp.einsum("bij,bj->bi", TG, zpgy)
    TGR_zpgy = jnp.einsum("bij,bj->bi", TGR, zpgy)
    TG_y = jnp.einsum("bij,bj->bi", TG, y_k)
    TGR_y = jnp.einsum("bij,bj->bi", TGR, y_k)

    z_new = (TG_zpgy - TGR_zpgy) + z_k - g_k[:, None] * y_k
    y_new = TG_y + TGR_y + y_k
    g_new = 2.0 * g_k

    # Solar source doubling
    R_sdown = jnp.einsum("bij,bj->bi", R_k, s_down_sol)
    R_sup = jnp.einsum("bij,bj->bi", R_k, s_up_sol)

    rhs_up = R_sdown + gamma_sol[:, None] * s_up_sol
    rhs_down = gamma_sol[:, None] * R_sup + s_down_sol

    s_up_sol_new = jnp.einsum("bij,bj->bi", TG, rhs_up) + s_up_sol
    s_down_sol_new = (jnp.einsum("bij,bj->bi", TG, rhs_down)
                      + gamma_sol[:, None] * s_down_sol)
    gamma_sol_new = gamma_sol * gamma_sol

    return (R_new, T_new, y_new, z_new, g_new,
            s_up_sol_new, s_down_sol_new, gamma_sol_new), None


def _add_layers(top, bot):
    """General adding, batched over wavenumbers.

    top/bot: tuples of (R_ab, R_ba, T_ab, T_ba, s_up, s_down, s_up_solar, s_down_solar)
    All matrices (nwav, N, N), vectors (nwav, N).
    """
    R_ab_t, R_ba_t, T_ab_t, T_ba_t, su_t, sd_t, sus_t, sds_t = top
    R_ab_b, R_ba_b, T_ab_b, T_ba_b, su_b, sd_b, sus_b, sds_b = bot

    N = R_ab_t.shape[-1]
    I_mat = jnp.eye(N)

    A1 = I_mat[None] - jnp.einsum("bij,bjk->bik", R_ab_b, R_ba_t)
    A2 = I_mat[None] - jnp.einsum("bij,bjk->bik", R_ba_t, R_ab_b)

    T_ba_D1 = _right_solve_batched(A1, T_ba_t)
    T_bc_D2 = _right_solve_batched(A2, T_ab_b)

    R_ab = R_ab_t + jnp.einsum("bij,bjk,bkl->bil", T_ba_D1, R_ab_b, T_ab_t)
    R_ba = R_ba_b + jnp.einsum("bij,bjk,bkl->bil", T_bc_D2, R_ba_t, T_ba_b)
    T_ab = jnp.einsum("bij,bjk->bik", T_bc_D2, T_ab_t)
    T_ba = jnp.einsum("bij,bjk->bik", T_ba_D1, T_ba_b)

    Rbc_sd = jnp.einsum("bij,bj->bi", R_ab_b, sd_t)
    s_up = su_t + jnp.einsum("bij,bj->bi", T_ba_D1, su_b + Rbc_sd)
    Rba_su = jnp.einsum("bij,bj->bi", R_ba_t, su_b)
    s_down = sd_b + jnp.einsum("bij,bj->bi", T_bc_D2, sd_t + Rba_su)

    Rbc_sds = jnp.einsum("bij,bj->bi", R_ab_b, sds_t)
    s_up_sol = sus_t + jnp.einsum("bij,bj->bi", T_ba_D1, sus_b + Rbc_sds)
    Rba_sus = jnp.einsum("bij,bj->bi", R_ba_t, sus_b)
    s_down_sol = sds_b + jnp.einsum("bij,bj->bi", T_bc_D2, sds_t + Rba_sus)

    return (R_ab, R_ba, T_ab, T_ba, s_up, s_down, s_up_sol, s_down_sol)


# ============================================================================
#  JIT-compiled core solver
# ============================================================================

@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _solve_core(nlay, N, nn_max, has_surface,
                delta_tau, ssa, planck_levels,
                PppC_all, PpmC_all,
                p_plus_solar_all, p_minus_solar_all,
                mu, wt, xfac, surface_albedo,
                solar_flux, solar_mu):
    """JIT-compiled core of the batch solver.

    Static args (traced as constants): nlay, N, nn_max, has_surface.
    Dynamic args (JAX arrays): everything else.

    delta_tau: (nwav, nlay)
    ssa: (nwav, nlay)
    planck_levels: (nwav, nlev)
    PppC_all: (nlay, N, N)
    PpmC_all: (nlay, N, N)
    p_plus_solar_all: (nlay, N)
    p_minus_solar_all: (nlay, N)
    mu, wt: (N,)
    solar_flux, solar_mu: scalars
    """
    nwav = delta_tau.shape[0]
    I_mat = jnp.eye(N)
    zero_mat = jnp.zeros((nwav, N, N))
    zero_vec = jnp.zeros((nwav, N))
    I_batch = jnp.broadcast_to(I_mat[None], (nwav, N, N))

    # Cumulative optical depth per wavenumber for solar attenuation
    tau_cum = jnp.concatenate([
        jnp.zeros((nwav, 1)),
        jnp.cumsum(delta_tau, axis=1),
    ], axis=1)  # (nwav, nlay+1)

    # --- 1. Doubling: compute per-layer R, T, sources ---

    def process_layer(carry, layer_idx):
        _ = carry
        tau_l = delta_tau[:, layer_idx]
        omega_l = ssa[:, layer_idx]
        B_top_l = planck_levels[:, layer_idx]
        B_bot_l = planck_levels[:, layer_idx + 1]

        B_bar = (B_bot_l + B_top_l) / 2.0
        B_d = jnp.where(tau_l > 0, (B_bot_l - B_top_l) / jnp.maximum(tau_l, 1e-30), 0.0)

        omega_clipped = jnp.clip(omega_l, 0.0, 1.0)
        con = 2.0 * omega_clipped * PI

        PppC = PppC_all[layer_idx]
        PpmC = PpmC_all[layer_idx]

        Gpp = (I_mat[None] - con[:, None, None] * PppC[None]) / mu[None, :, None]
        Gpm = con[:, None, None] * PpmC[None] / mu[None, :, None]

        tau0 = tau_l / (2.0 ** nn_max)

        # Initial thermal state
        R_k = tau0[:, None, None] * Gpm
        T_k = I_mat[None] - tau0[:, None, None] * Gpp
        y_k = (1.0 - omega_clipped)[:, None] * tau0[:, None] / mu[None, :]
        z_k = jnp.zeros_like(y_k)
        g_k = 0.5 * tau0

        # Initial solar state
        tau_cum_l = tau_cum[:, layer_idx]
        F_top = solar_flux * jnp.exp(-tau_cum_l / jnp.maximum(solar_mu, 1e-30))
        p_plus = p_plus_solar_all[layer_idx]    # (N,)
        p_minus = p_minus_solar_all[layer_idx]   # (N,)

        base = omega_clipped[:, None] * tau0[:, None] / mu[None, :] * F_top[:, None]
        s_up_sol = base * p_minus[None, :]
        s_down_sol = base * p_plus[None, :]
        gamma_sol = jnp.exp(-tau0 / jnp.maximum(solar_mu, 1e-30))

        # Doubling iterations
        (R_k, T_k, y_k, z_k, _, s_up_sol, s_down_sol, _), _ = jax.lax.scan(
            _doubling_step,
            (R_k, T_k, y_k, z_k, g_k, s_up_sol, s_down_sol, gamma_sol),
            None, length=nn_max)

        s_up = y_k * B_bar[:, None] + z_k * B_d[:, None]
        s_down = y_k * B_bar[:, None] - z_k * B_d[:, None]

        layer_data = (R_k, R_k, T_k, T_k, s_up, s_down, s_up_sol, s_down_sol)
        return None, layer_data

    _, all_layers = jax.lax.scan(
        process_layer, None, jnp.arange(nlay))

    # --- 2. Surface layer ---
    def make_surface():
        A = surface_albedo
        R_surf_row = 2.0 * A * (mu * wt) * xfac
        R_surf = jnp.broadcast_to(R_surf_row[None, None, :], (nwav, N, N))
        B_surface = planck_levels[:, -1]

        s_up_surf = ((1.0 - A) * B_surface)[:, None] * jnp.ones((1, N))

        # Solar reflection from surface
        tau_total = tau_cum[:, -1]
        solar_at_surface = (A / PI) * solar_flux * solar_mu * jnp.exp(
            -tau_total / jnp.maximum(solar_mu, 1e-30))
        s_up_solar_surf = solar_at_surface[:, None] * jnp.ones((1, N))

        return (
            R_surf, R_surf,
            zero_mat, zero_mat,
            s_up_surf, zero_vec,
            s_up_solar_surf, zero_vec,
        )

    surface = make_surface()

    # --- 3. Build composites from bottom (RBASE) using scan ---
    ltot = nlay + 1 if has_surface else nlay

    if has_surface:
        rbase_composite = surface
    else:
        rbase_composite = tuple(a[nlay - 1] for a in all_layers)

    def rbase_step(composite, k):
        if has_surface:
            layer_idx = nlay - 1 - k
        else:
            layer_idx = nlay - 2 - k
        layer = tuple(a[layer_idx] for a in all_layers)
        new_composite = _add_layers(layer, composite)
        return new_composite, new_composite

    n_rbase_steps = ltot - 1
    if n_rbase_steps > 0:
        rbase_full, rbase_all = jax.lax.scan(
            rbase_step, rbase_composite, jnp.arange(n_rbase_steps))
    else:
        rbase_full = rbase_composite

    # --- 4. Build composites from top (RTOP) using scan ---
    rtop_composite = tuple(a[0] for a in all_layers)

    def rtop_step(composite, k):
        layer_idx = k + 1
        layer = tuple(a[layer_idx] for a in all_layers)
        new_composite = _add_layers(composite, layer)
        return new_composite, new_composite

    n_rtop_steps = nlay - 1
    if n_rtop_steps > 0:
        rtop_at_nlay, rtop_all = jax.lax.scan(
            rtop_step, rtop_composite, jnp.arange(n_rtop_steps))
    else:
        rtop_at_nlay = rtop_composite

    # --- 5. Compute fluxes at TOA ---
    Iup_toa = rbase_full[4] + rbase_full[6]  # s_up + s_up_solar
    flux_up_toa = jnp.sum(2.0 * PI * wt[None, :] * mu[None, :] * Iup_toa, axis=1)

    # --- 6. Compute fluxes at BOA ---
    if has_surface:
        base_at_boa = surface
        top_c = rtop_at_nlay
        base_c = base_at_boa
        to_inv = I_batch - jnp.einsum("bij,bjk->bik", top_c[1], base_c[0])
        rhs = (top_c[5] + top_c[7]
               + jnp.einsum("bij,bj->bi", top_c[1], base_c[4] + base_c[6]))
        Idown_boa = _batch_solve_vec(to_inv, rhs)
    else:
        Idown_boa = rtop_at_nlay[5] + rtop_at_nlay[7]

    flux_down_boa = jnp.sum(2.0 * PI * wt[None, :] * mu[None, :] * Idown_boa, axis=1)

    return flux_up_toa, flux_down_boa


# ============================================================================
#  Public API
# ============================================================================

class BatchConfig:
    """Configuration for the batched solver."""
    def __init__(self):
        self.num_wavenumbers = 0
        self.num_layers = 0
        self.num_quadrature = 8
        self.num_moments_max = 16
        self.surface_albedo = 0.0
        self.solar_flux = 0.0
        self.solar_mu = 1.0


def _compute_nn_max(delta_tau, ssa):
    """Compute max doubling iterations across all wavenumbers and layers."""
    tau_flat = np.asarray(delta_tau).ravel()
    omega_flat = np.asarray(ssa).ravel()
    # Filter to scattering entries only
    mask = omega_flat > 0.0
    if not np.any(mask):
        return 1
    tau_s = tau_flat[mask]
    omega_s = omega_flat[mask]
    ipow0 = np.where(omega_s < 0.01, 4, np.where(omega_s < 0.1, 10, 16))
    nn = np.maximum(1, (np.log(np.maximum(tau_s, 1e-30)) / np.log(2.0)).astype(int) + ipow0)
    return int(np.max(nn))


def solve_batch(config, delta_tau, ssa, phase_moments, planck_levels):
    """Solve the RT problem for a batch of wavenumbers.

    Args:
        config: BatchConfig instance.
        delta_tau: (nwav, nlay) optical depths.
        ssa: (nwav, nlay) single-scattering albedos.
        phase_moments: (nlay, nmom) Legendre moments (shared across wavenumbers).
        planck_levels: (nwav, nlev) Planck values at level interfaces.

    Returns:
        (flux_up_toa, flux_down_boa): arrays of shape (nwav,).
    """
    nlay = config.num_layers
    N = config.num_quadrature
    nmom = config.num_moments_max

    # Precompute (outside JIT): quadrature, phase matrices, max doubling iters
    mu, wt = gauss_legendre(N)
    xfac_sum = jnp.sum(mu * wt)
    xfac = 0.5 / xfac_sum

    Pl = precompute_legendre_polynomials(nmom, mu)
    C = jnp.diag(wt)

    has_solar = config.solar_flux > 0.0 and config.solar_mu > 0.0

    PppC_list = []
    PpmC_list = []
    p_plus_list = []
    p_minus_list = []
    for l in range(nlay):
        chi = jnp.asarray(phase_moments[l])
        Ppp, Ppm = compute_phase_matrices(chi, mu, wt, Pl)
        PppC_list.append(Ppp @ C)
        PpmC_list.append(Ppm @ C)

        if has_solar:
            pp, pm = compute_solar_phase_vectors(
                chi, mu, wt, config.solar_mu, Pl)
            p_plus_list.append(pp)
            p_minus_list.append(pm)
        else:
            p_plus_list.append(jnp.zeros(N))
            p_minus_list.append(jnp.zeros(N))

    PppC_all = jnp.stack(PppC_list)    # (nlay, N, N)
    PpmC_all = jnp.stack(PpmC_list)    # (nlay, N, N)
    p_plus_all = jnp.stack(p_plus_list)   # (nlay, N)
    p_minus_all = jnp.stack(p_minus_list) # (nlay, N)

    nn_max = _compute_nn_max(delta_tau, ssa)
    has_surface = config.surface_albedo > 0.0 or float(np.max(planck_levels[:, -1])) > 0.0

    return _solve_core(
        nlay, N, nn_max, has_surface,
        jnp.asarray(delta_tau),
        jnp.asarray(ssa),
        jnp.asarray(planck_levels),
        PppC_all, PpmC_all,
        p_plus_all, p_minus_all,
        mu, wt, xfac,
        config.surface_albedo,
        config.solar_flux,
        config.solar_mu,
    )
