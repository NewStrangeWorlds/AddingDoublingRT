"""Main solver: implements the full adding-doubling radiative transfer method."""

import copy

import jax.numpy as jnp
import numpy as np

from .adding import add_layers
from .config import ADConfig, RTOutput
from .doubling import doubling
from .phase_matrix import compute_phase_matrices, compute_solar_phase_vectors
from .planck import planck_function
from .quadrature import gauss_legendre, precompute_legendre_polynomials

PI = jnp.pi


def _empty_layer(N):
    """Create an empty layer (identity transmission, zero reflection/sources)."""
    return {
        "R_ab": jnp.zeros((N, N)),
        "R_ba": jnp.zeros((N, N)),
        "T_ab": jnp.eye(N),
        "T_ba": jnp.eye(N),
        "s_up": jnp.zeros(N),
        "s_down": jnp.zeros(N),
        "s_up_solar": jnp.zeros(N),
        "s_down_solar": jnp.zeros(N),
        "is_scattering": False,
    }


def solve(config):
    """Solve the radiative transfer problem.

    Args:
        config: ADConfig instance with solver configuration.

    Returns:
        RTOutput with fluxes and mean intensities at each layer interface.
    """
    cfg = copy.deepcopy(config)
    cfg.validate()

    nlay = cfg.num_layers
    N = cfg.num_quadrature

    # Reverse arrays if indexed from bottom
    if cfg.index_from_bottom:
        cfg.delta_tau = cfg.delta_tau[::-1].copy()
        cfg.single_scat_albedo = cfg.single_scat_albedo[::-1].copy()
        cfg.phase_function_moments = cfg.phase_function_moments[::-1]
        if cfg.use_thermal_emission:
            cfg.temperature = cfg.temperature[::-1].copy()
        if cfg.planck_levels is not None:
            cfg.planck_levels = cfg.planck_levels[::-1].copy()

    # 1. Gauss-Legendre quadrature on [0, 1]
    mu, wt = gauss_legendre(N)

    xfac_sum = jnp.sum(mu * wt)
    xfac = 0.5 / xfac_sum

    # 2. Compute Planck values at each level
    B = np.zeros(nlay + 1)
    B_surface = 0.0
    B_top_emission = 0.0

    if cfg.use_thermal_emission:
        for l in range(nlay + 1):
            B[l] = planck_function(cfg.wavenumber_low, cfg.wavenumber_high, cfg.temperature[l])
        B_surface = B[nlay]
        B_top_emission = B[0]
    elif cfg.planck_levels is not None and len(cfg.planck_levels) == nlay + 1:
        B = np.array(cfg.planck_levels)
        B_surface = cfg.surface_emission
        B_top_emission = cfg.top_emission
    else:
        B_surface = cfg.surface_emission
        B_top_emission = cfg.top_emission

    # 3. Compute per-layer R, T, s
    has_solar = cfg.solar_flux > 0.0 and cfg.solar_mu > 0.0
    two_M = 2 * N

    layer_rtj = []
    tau_used = np.zeros(nlay)

    # Cache for Legendre polynomials
    _Pl_cache = {}

    def get_legendre(L_val):
        if L_val not in _Pl_cache:
            _Pl_cache[L_val] = precompute_legendre_polynomials(L_val, mu)
        return _Pl_cache[L_val]

    tau_cumulative = 0.0
    for l in range(nlay):
        tau_layer = float(cfg.delta_tau[l])
        omega_layer = float(cfg.single_scat_albedo[l])
        B_layer_top = B[l]
        B_layer_bot = B[l + 1]

        Ppp = jnp.zeros((N, N))
        Ppm = jnp.zeros((N, N))
        p_plus_solar = None
        p_minus_solar = None

        if omega_layer > 0.0 and tau_layer > 0.0:
            chi_full = np.array(cfg.phase_function_moments[l])

            if cfg.use_delta_m:
                f_trunc = chi_full[two_M] if len(chi_full) > two_M else 0.0

                if f_trunc > 1e-12 and f_trunc < 1.0 - 1e-12:
                    omega_f = omega_layer * f_trunc
                    tau_layer = (1.0 - omega_f) * float(cfg.delta_tau[l])
                    omega_layer = omega_layer * (1.0 - f_trunc) / (1.0 - omega_f)

                    chi_star = np.array([(chi_full[ll] - f_trunc) / (1.0 - f_trunc)
                                         for ll in range(two_M)])
                    Pl = get_legendre(two_M)
                    Ppp, Ppm = compute_phase_matrices(chi_star, mu, wt, Pl)
                    if has_solar:
                        p_plus_solar, p_minus_solar = compute_solar_phase_vectors(
                            chi_star, mu, wt, cfg.solar_mu, Pl)
                else:
                    chi = chi_full[:min(len(chi_full), two_M)]
                    Pl = get_legendre(len(chi))
                    Ppp, Ppm = compute_phase_matrices(chi, mu, wt, Pl)
                    if has_solar:
                        p_plus_solar, p_minus_solar = compute_solar_phase_vectors(
                            chi, mu, wt, cfg.solar_mu, Pl)
            else:
                Pl = get_legendre(len(chi_full))
                Ppp, Ppm = compute_phase_matrices(chi_full, mu, wt, Pl)
                if has_solar:
                    p_plus_solar, p_minus_solar = compute_solar_phase_vectors(
                        chi_full, mu, wt, cfg.solar_mu, Pl)

        tau_used[l] = tau_layer

        layer_rtj.append(
            doubling(tau_layer, omega_layer, B_layer_top, B_layer_bot,
                     Ppp, Ppm, mu, wt,
                     cfg.solar_flux, cfg.solar_mu, tau_cumulative,
                     p_plus_solar, p_minus_solar)
        )
        tau_cumulative += tau_layer

    # 4. Lambertian surface layer
    ltot = nlay
    has_surface = (not cfg.use_diffusion_lower_bc
                   and (cfg.surface_albedo > 0.0 or B_surface > 0.0))

    if has_surface:
        A = cfg.surface_albedo
        R_surf = 2.0 * A * (mu * wt)[None, :] * xfac
        R_surf = jnp.broadcast_to(R_surf, (N, N))

        s_up_surf = jnp.full(N, (1.0 - A) * B_surface)
        s_up_solar_surf = jnp.zeros(N)
        if has_solar and A > 0.0:
            s_up_solar_surf = jnp.full(
                N, (A / PI) * cfg.solar_flux * cfg.solar_mu
                   * jnp.exp(-tau_cumulative / cfg.solar_mu)
            )

        surf = {
            "R_ab": R_surf,
            "R_ba": R_surf,
            "T_ab": jnp.zeros((N, N)),
            "T_ba": jnp.zeros((N, N)),
            "s_up": s_up_surf,
            "s_down": jnp.zeros(N),
            "s_up_solar": s_up_solar_surf,
            "s_down_solar": jnp.zeros(N),
            "is_scattering": A > 0.0,
        }
        layer_rtj.append(surf)
        ltot += 1

    # 5. Build composites from bottom (RBASE)
    rbase = [_empty_layer(N)]
    rbase.append(layer_rtj[ltot - 1])

    for l in range(1, ltot):
        k = ltot - 1 - l
        rbase.append(add_layers(layer_rtj[k], rbase[l]))

    # 6. Build composites from top (RTOP)
    rtop = [_empty_layer(N)]
    rtop.append(layer_rtj[0])

    for l in range(1, ltot):
        rtop.append(add_layers(rtop[l], layer_rtj[l]))

    # 7. Boundary intensities
    I_top_down = jnp.full(N, B_top_emission)
    I_bot_up = jnp.zeros(N)

    if cfg.use_diffusion_lower_bc:
        B_bottom = B[nlay]
        dtau_last = tau_used[nlay - 1]
        dB_dtau = (B_bottom - B[nlay - 1]) / dtau_last if dtau_last > 0.0 else 0.0
        I_bot_up = B_bottom + mu * dB_dtau
    elif not has_surface:
        I_bot_up = jnp.full(N, B_surface)

    # 8. Compute intensities at each interface
    n_interfaces = nlay + 1
    flux_up = np.zeros(n_interfaces)
    flux_down = np.zeros(n_interfaces)
    mean_intensity = np.zeros(n_interfaces)
    flux_direct = np.zeros(n_interfaces)

    if has_solar:
        tau_cum = 0.0
        flux_direct[0] = cfg.solar_flux * cfg.solar_mu
        for l in range(nlay):
            tau_cum += tau_used[l]
            flux_direct[l + 1] = cfg.solar_flux * cfg.solar_mu * np.exp(-tau_cum / cfg.solar_mu)

    # Top of atmosphere
    full = rbase[ltot]
    Iup = (full["R_ab"] @ I_top_down + full["T_ba"] @ I_bot_up
           + full["s_up"] + full["s_up_solar"])

    flux_up[0] = float(jnp.sum(2.0 * PI * wt * mu * Iup))
    flux_down[0] = float(jnp.sum(2.0 * PI * wt * mu * I_top_down))
    mean_intensity[0] = float(jnp.sum(0.5 * wt * (Iup + I_top_down)))

    # Internal interfaces
    for l in range(1, nlay + 1):
        n_top = l
        n_base = ltot - l

        if n_base > 0 and n_top > 0:
            top_c = rtop[n_top]
            base_c = rbase[n_base]
            I_mat = jnp.eye(N)

            # Compute I_up
            to_inv = I_mat - base_c["R_ab"] @ top_c["R_ba"]
            rhs = (base_c["T_ba"] @ I_bot_up
                   + base_c["R_ab"] @ (top_c["T_ab"] @ I_top_down
                                       + top_c["s_down"] + top_c["s_down_solar"])
                   + base_c["s_up"] + base_c["s_up_solar"])
            Iup = jnp.linalg.solve(to_inv, rhs)

            # Compute I_down
            to_inv = I_mat - top_c["R_ba"] @ base_c["R_ab"]
            rhs = (top_c["T_ab"] @ I_top_down
                   + top_c["R_ba"] @ (base_c["T_ba"] @ I_bot_up
                                      + base_c["s_up"] + base_c["s_up_solar"])
                   + top_c["s_down"] + top_c["s_down_solar"])
            Idown = jnp.linalg.solve(to_inv, rhs)

        elif n_base == 0:
            Idown = (rtop[n_top]["T_ab"] @ I_top_down
                     + rtop[n_top]["R_ba"] @ I_bot_up
                     + rtop[n_top]["s_down"] + rtop[n_top]["s_down_solar"])
            Iup = I_bot_up
        else:
            Iup = jnp.zeros(N)
            Idown = I_top_down

        flux_up[l] = float(jnp.sum(2.0 * PI * wt * mu * Iup))
        flux_down[l] = float(jnp.sum(2.0 * PI * wt * mu * Idown))
        mean_intensity[l] = float(jnp.sum(0.5 * wt * (Iup + Idown)))

    # Reverse output if indexed from bottom
    if cfg.index_from_bottom:
        flux_up = flux_up[::-1]
        flux_down = flux_down[::-1]
        mean_intensity = mean_intensity[::-1]
        flux_direct = flux_direct[::-1]

    return RTOutput(
        flux_up=jnp.array(flux_up),
        flux_down=jnp.array(flux_down),
        mean_intensity=jnp.array(mean_intensity),
        flux_direct=jnp.array(flux_direct),
    )
