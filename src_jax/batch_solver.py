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
from .doubling import doubling_start

PI = jnp.pi


def _batch_solve_vec(A, b):
    """Batched solve A x = b. Float32 interface, float64 LU for stability."""
    x = jnp.linalg.solve(A.astype(jnp.float64), b[..., None].astype(jnp.float64))
    return x.squeeze(-1).astype(jnp.float32)


def _right_solve_batched(A, B):
    """Batched right solve: X A = B. Float32 interface, float64 LU for stability."""
    AT = jnp.swapaxes(A, -2, -1).astype(jnp.float64)
    BT = jnp.swapaxes(B, -2, -1).astype(jnp.float64)
    return jnp.linalg.solve(AT, BT).swapaxes(-2, -1).astype(jnp.float32)


# ============================================================================
#  Core operations (all JIT-compatible, no Python control flow)
# ============================================================================

def _doubling_step(carry, _):
    """Single doubling iteration. Used inside jax.lax.scan."""
    R_k, T_k, y_k, z_k, g_k, s_up_sol, s_down_sol, gamma_sol = carry
    N = R_k.shape[-1]
    I_mat = jnp.eye(N, dtype=jnp.float32)

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
    I_mat = jnp.eye(N, dtype=jnp.float32)

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

@functools.partial(jax.jit, static_argnums=(0, 1, 3))
def _solve_core(nlay, N, nn_layers, has_surface,
                delta_tau, ssa, planck_levels,
                PppC_all, PpmC_all,
                p_plus_solar_all, p_minus_solar_all,
                mu, wt, xfac, surface_albedo,
                solar_flux, solar_mu):
    """JIT-compiled core of the batch solver.

    Static args (traced as constants): nlay, N, has_surface.
    Dynamic args (JAX arrays): everything else; nn_layers (nlay,) int32 is the
    per-layer doubling count (max over the wavenumbers of that layer).

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
    I_mat = jnp.eye(N, dtype=jnp.float32)
    zero_mat = jnp.zeros((nwav, N, N), dtype=jnp.float32)
    zero_vec = jnp.zeros((nwav, N), dtype=jnp.float32)
    I_batch = jnp.broadcast_to(I_mat[None], (nwav, N, N))

    # Cumulative optical depth per wavenumber for solar attenuation
    tau_cum = jnp.concatenate([
        jnp.zeros((nwav, 1), dtype=jnp.float32),
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

        Spp = con[:, None, None] * PppC[None] / mu[None, :, None]
        Spm = con[:, None, None] * PpmC[None] / mu[None, :, None]

        nn_l = nn_layers[layer_idx]
        tau0 = tau_l / jnp.exp2(nn_l.astype(jnp.float32))
        g_k = 0.5 * tau0

        # Initial state (exact extinction + single scattering, Taylor double scattering)
        tau_cum_l = tau_cum[:, layer_idx]
        F_top = solar_flux * jnp.exp(-tau_cum_l / jnp.maximum(solar_mu, 1e-30))
        p_plus = p_plus_solar_all[layer_idx]    # (N,)
        p_minus = p_minus_solar_all[layer_idx]   # (N,)

        R_k, T_k, y_k, z_k, s_up_sol, s_down_sol = doubling_start(
            tau0, omega_clipped, Spp, Spm, mu,
            F_top, p_plus, p_minus, jnp.maximum(solar_mu, 1e-30))
        gamma_sol = jnp.exp(-tau0 / jnp.maximum(solar_mu, 1e-30))

        # Doubling iterations: per-layer trip count (dynamic -> while loop)
        (R_k, T_k, y_k, z_k, _, s_up_sol, s_down_sol, _) = jax.lax.fori_loop(
            0, nn_l,
            lambda _k, c: _doubling_step(c, None)[0],
            (R_k, T_k, y_k, z_k, g_k, s_up_sol, s_down_sol, gamma_sol))

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

        s_up_surf = ((1.0 - A) * B_surface)[:, None] * jnp.ones((1, N), dtype=jnp.float32)

        # Solar reflection from surface
        tau_total = tau_cum[:, -1]
        solar_at_surface = (A / PI) * solar_flux * solar_mu * jnp.exp(
            -tau_total / jnp.maximum(solar_mu, 1e-30))
        s_up_solar_surf = solar_at_surface[:, None] * jnp.ones((1, N), dtype=jnp.float32)

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
#  Sequential-wavenumber core (CPU-efficient via jax.lax.map)
# ============================================================================

@functools.partial(jax.jit, static_argnums=(0, 1, 3))
def _solve_core_map(nlay, N, nn_wl, has_surface,
                    delta_tau, ssa, planck_levels,
                    PppC_all, PpmC_all,
                    p_plus_solar_all, p_minus_solar_all,
                    mu, wt, xfac, surface_albedo,
                    solar_flux, solar_mu):
    """RT solve using jax.lax.map over wavenumbers.

    Processes one wavenumber at a time so the working set (a few N×N matrices)
    stays in L1 cache. Preferred on CPU; the batched _solve_core is faster on GPU.
    """
    I_mat    = jnp.eye(N, dtype=jnp.float32)
    zero_mat = jnp.zeros((N, N), dtype=jnp.float32)
    zero_vec = jnp.zeros(N, dtype=jnp.float32)

    def _right_solve_1(A, B):
        """Right solve X A = B for single N×N. Float32 in/out, float64 LU."""
        AT = A.T.astype(jnp.float64)
        BT = B.T.astype(jnp.float64)
        return jnp.linalg.solve(AT, BT).T.astype(jnp.float32)

    def _solve_vec_1(A, b):
        """Solve A x = b for single N×N. Float32 in/out, float64 LU."""
        return jnp.linalg.solve(
            A.astype(jnp.float64), b.astype(jnp.float64)
        ).astype(jnp.float32)

    def _add_layers_1(top, bot):
        R_ab_t, R_ba_t, T_ab_t, T_ba_t, su_t, sd_t, sus_t, sds_t = top
        R_ab_b, R_ba_b, T_ab_b, T_ba_b, su_b, sd_b, sus_b, sds_b = bot

        A1 = I_mat - R_ab_b @ R_ba_t
        A2 = I_mat - R_ba_t @ R_ab_b
        T_ba_D1 = _right_solve_1(A1, T_ba_t)
        T_bc_D2 = _right_solve_1(A2, T_ab_b)

        R_ab = R_ab_t + T_ba_D1 @ R_ab_b @ T_ab_t
        R_ba = R_ba_b + T_bc_D2 @ R_ba_t @ T_ba_b
        T_ab = T_bc_D2 @ T_ab_t
        T_ba = T_ba_D1 @ T_ba_b

        Rbc_sd  = R_ab_b @ sd_t
        s_up    = su_t  + T_ba_D1 @ (su_b  + Rbc_sd)
        Rba_su  = R_ba_t @ su_b
        s_down  = sd_b  + T_bc_D2 @ (sd_t  + Rba_su)

        Rbc_sds   = R_ab_b @ sds_t
        s_up_sol  = sus_t + T_ba_D1 @ (sus_b  + Rbc_sds)
        Rba_sus   = R_ba_t @ sus_b
        s_down_sol = sds_b + T_bc_D2 @ (sds_t + Rba_sus)

        return (R_ab, R_ba, T_ab, T_ba, s_up, s_down, s_up_sol, s_down_sol)

    def solve_one(wav_inputs):
        """Full RT solve for a single wavenumber — all arrays are unbatched."""
        dtau_1, ssa_1, plan_1, nn_1 = wav_inputs   # (nlay,), (nlay,), (nlev,), (nlay,) int32

        tau_cum_1 = jnp.concatenate([
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.cumsum(dtau_1),
        ])  # (nlay+1,)

        # --- Doubling: one layer at a time ---
        def process_layer_1(carry, layer_idx):
            tau_l   = dtau_1[layer_idx]
            omega_l = ssa_1[layer_idx]
            B_top_l = plan_1[layer_idx]
            B_bot_l = plan_1[layer_idx + 1]

            B_bar = (B_bot_l + B_top_l) / 2.0
            B_d   = jnp.where(tau_l > 0,
                               (B_bot_l - B_top_l) / jnp.maximum(tau_l, 1e-30),
                               0.0)

            omega_c = jnp.clip(omega_l, 0.0, 1.0)
            con = 2.0 * omega_c * PI

            Spp = con * PppC_all[layer_idx] / mu[:, None]
            Spm = con * PpmC_all[layer_idx] / mu[:, None]

            nn_l = nn_1[layer_idx]
            tau0 = tau_l / jnp.exp2(nn_l.astype(jnp.float32))
            g_k  = 0.5 * tau0   # scalar

            F_top = solar_flux * jnp.exp(
                -tau_cum_1[layer_idx] / jnp.maximum(solar_mu, 1e-30))
            R_k, T_k, y_k, z_k, s_up_sol, s_down_sol = doubling_start(
                tau0, omega_c, Spp, Spm, mu,
                F_top, p_plus_solar_all[layer_idx], p_minus_solar_all[layer_idx],
                jnp.maximum(solar_mu, 1e-30))
            gamma_sol  = jnp.exp(-tau0 / jnp.maximum(solar_mu, 1e-30))  # scalar

            def doubling_1(carry, _):
                R, T, y, z, g, su_s, sd_s, gam = carry

                TG  = _right_solve_1(I_mat - R @ R, T)
                TGR = TG @ R

                R_new = R + TGR @ T
                T_new = TG @ T

                zpgy  = z + g * y
                z_new = (TG @ zpgy - TGR @ zpgy) + z - g * y
                y_new = TG @ y + TGR @ y + y
                g_new = 2.0 * g

                rhs_up   = R @ sd_s + gam * su_s
                rhs_down = gam * (R @ su_s) + sd_s
                su_s_new = TG @ rhs_up  + su_s
                sd_s_new = TG @ rhs_down + gam * sd_s

                return (R_new, T_new, y_new, z_new, g_new,
                        su_s_new, sd_s_new, gam * gam), None

            (R_k, T_k, y_k, z_k, _, s_up_sol, s_down_sol, _) = jax.lax.fori_loop(
                0, nn_l,
                lambda _k, c: doubling_1(c, None)[0],
                (R_k, T_k, y_k, z_k, g_k, s_up_sol, s_down_sol, gamma_sol))

            s_up   = y_k * B_bar + z_k * B_d
            s_down = y_k * B_bar - z_k * B_d
            return None, (R_k, R_k, T_k, T_k, s_up, s_down, s_up_sol, s_down_sol)

        _, all_layers = jax.lax.scan(process_layer_1, None, jnp.arange(nlay))

        # --- Surface layer ---
        A_s = surface_albedo
        R_surf = jnp.broadcast_to((2.0 * A_s * mu * wt * xfac)[None, :], (N, N))
        s_up_surf     = jnp.full(N, (1.0 - A_s) * plan_1[-1], dtype=jnp.float32)
        solar_at_surf = ((A_s / PI) * solar_flux * solar_mu
                         * jnp.exp(-tau_cum_1[-1] / jnp.maximum(solar_mu, 1e-30)))
        s_up_sol_surf = jnp.full(N, solar_at_surf, dtype=jnp.float32)
        surface = (R_surf, R_surf, zero_mat, zero_mat,
                   s_up_surf, zero_vec, s_up_sol_surf, zero_vec)

        # --- RBASE: accumulate from bottom ---
        ltot = nlay + 1 if has_surface else nlay
        rbase_init = surface if has_surface else tuple(a[nlay - 1] for a in all_layers)

        def rbase_step_1(comp, k):
            idx = (nlay - 1 - k) if has_surface else (nlay - 2 - k)
            return _add_layers_1(tuple(a[idx] for a in all_layers), comp), None

        rbase_full, _ = jax.lax.scan(rbase_step_1, rbase_init, jnp.arange(ltot - 1))

        # --- RTOP: accumulate from top ---
        rtop_init = tuple(a[0] for a in all_layers)

        def rtop_step_1(comp, k):
            return _add_layers_1(comp, tuple(a[k + 1] for a in all_layers)), None

        rtop_nlay, _ = jax.lax.scan(rtop_step_1, rtop_init, jnp.arange(nlay - 1))

        # --- TOA flux ---
        flux_up = jnp.sum(2.0 * PI * wt * mu * (rbase_full[4] + rbase_full[6]))

        # --- BOA flux ---
        if has_surface:
            to_inv    = I_mat - rtop_nlay[1] @ surface[0]
            rhs_b     = rtop_nlay[5] + rtop_nlay[7] + rtop_nlay[1] @ (surface[4] + surface[6])
            Idown_boa = _solve_vec_1(to_inv, rhs_b)
        else:
            Idown_boa = rtop_nlay[5] + rtop_nlay[7]

        flux_down = jnp.sum(2.0 * PI * wt * mu * Idown_boa)
        return flux_up, flux_down

    return jax.lax.map(solve_one, (delta_tau, ssa, planck_levels, nn_wl))


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


def _compute_nn_layers(delta_tau, ssa, mu_min):
    """Per-(wavenumber, layer) doubling counts, shape (nwav, nlay), int32.

    Omega-adaptive rule plus the extinction floor tau0 = tau / 2**nn <= mu_min / 2
    (mirrors adrt::computeDoublingCount). Non-scattering or empty entries get 1:
    the thin-layer start is exact for omega = 0 at any tau0.
    """
    tau = np.asarray(delta_tau, dtype=np.float64)
    omega = np.asarray(ssa, dtype=np.float64)
    mask = (omega > 0.0) & (tau > 0.0)
    tau_s = np.where(mask, tau, 1.0)
    ipow0 = np.where(omega < 0.01, 4, np.where(omega < 0.1, 10, 16))
    nn = np.floor(np.log2(tau_s)).astype(int) + ipow0
    n_ext = np.ceil(np.log2(tau_s / mu_min)).astype(int) + 1
    nn = np.maximum(1, np.maximum(nn, n_ext))
    return np.where(mask, nn, 1).astype(np.int32)


def _compute_nn_max(delta_tau, ssa, mu_min):
    """Max doubling count over all wavenumbers and layers (diagnostics)."""
    return int(np.max(_compute_nn_layers(delta_tau, ssa, mu_min)))


def solve_batch(config, delta_tau, ssa, phase_moments, planck_levels, use_map=False):
    """Solve the RT problem for a batch of wavenumbers.

    Args:
        config: BatchConfig instance.
        delta_tau: (nwav, nlay) optical depths.
        ssa: (nwav, nlay) single-scattering albedos.
        phase_moments: (nlay, nmom) Legendre moments (shared across wavenumbers).
        planck_levels: (nwav, nlev) Planck values at level interfaces.
        use_map: If True, use jax.lax.map to process one wavenumber at a time
                 (cache-friendly on CPU). Default False uses the batched kernel
                 (better on GPU).

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

    PppC_all = jnp.stack(PppC_list).astype(jnp.float32)    # (nlay, N, N)
    PpmC_all = jnp.stack(PpmC_list).astype(jnp.float32)    # (nlay, N, N)
    p_plus_all = jnp.stack(p_plus_list).astype(jnp.float32)   # (nlay, N)
    p_minus_all = jnp.stack(p_minus_list).astype(jnp.float32) # (nlay, N)

    # Per-layer doubling counts: the batched core runs each layer with the max
    # over its wavenumbers; the map core uses the per-wavenumber values.
    nn_wl = _compute_nn_layers(delta_tau, ssa, float(np.min(np.asarray(mu))))
    nn_arg = jnp.asarray(nn_wl if use_map else nn_wl.max(axis=0), dtype=jnp.int32)
    has_surface = config.surface_albedo > 0.0 or float(np.max(planck_levels[:, -1])) > 0.0

    mu = mu.astype(jnp.float32)
    wt = wt.astype(jnp.float32)
    xfac = jnp.float32(xfac)

    core = _solve_core_map if use_map else _solve_core
    return core(
        nlay, N, nn_arg, has_surface,
        jnp.asarray(delta_tau, dtype=jnp.float32),
        jnp.asarray(ssa, dtype=jnp.float32),
        jnp.asarray(planck_levels, dtype=jnp.float32),
        PppC_all, PpmC_all,
        p_plus_all, p_minus_all,
        mu, wt, xfac,
        config.surface_albedo,
        config.solar_flux,
        config.solar_mu,
    )
