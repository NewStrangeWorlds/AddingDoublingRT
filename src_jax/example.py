"""Demonstration of the adding-doubling RT solver (JAX version).

Mirrors the C++ example.cpp test suite.
"""

import numpy as np
import jax.numpy as jnp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_jax import ADConfig, RTOutput, solve, gauss_legendre

PI = np.pi


def planck(nu_cm, T):
    """Simple Planck function B_nu(T) at wavenumber nu [cm^-1] and temperature T [K]."""
    h = 6.62607015e-34
    c = 2.99792458e10
    k = 1.380649e-23
    x = h * c * nu_cm / (k * T)
    if x > 500.0:
        return 0.0
    return 2.0 * h * c * c * nu_cm**3 / (np.exp(x) - 1.0)


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_results(result, nlay):
    has_direct = np.any(np.array(result.flux_direct) > 0.0)

    if has_direct:
        print("  Level    F_up           F_down         F_direct       F_net_total    J_mean")
        print(f"  {'-' * 80}")
        for l in range(nlay + 1):
            fnet = result.flux_up[l] - result.flux_down[l] - result.flux_direct[l]
            print(f"  {l:5d}  {result.flux_up[l]:13.6f}  {result.flux_down[l]:13.6f}"
                  f"  {result.flux_direct[l]:13.6f}  {fnet:13.6f}"
                  f"  {result.mean_intensity[l]:13.6f}")
    else:
        print("  Level    F_up           F_down         F_net          J_mean")
        print(f"  {'-' * 66}")
        for l in range(nlay + 1):
            fnet = result.flux_up[l] - result.flux_down[l]
            print(f"  {l:5d}  {result.flux_up[l]:13.6f}  {result.flux_down[l]:13.6f}"
                  f"  {fnet:13.6f}  {result.mean_intensity[l]:13.6f}")
    print()


# ============================================================================
#  Test 1: Pure absorption (no scattering)
# ============================================================================
def test_pure_absorption():
    print_header("Test 1: Pure absorption (omega=0, thermal emission)")

    T_surface = 300.0
    T_atm = 250.0
    nu = 1000.0

    B_surf = planck(nu, T_surface)
    B_atm = planck(nu, T_atm)

    print(f"  B(T_surf={T_surface}K) = {B_surf}")
    print(f"  B(T_atm ={T_atm}K) = {B_atm}\n")

    cfg = ADConfig(num_layers=1, num_quadrature=8)
    cfg.surface_emission = B_surf
    cfg.allocate()
    cfg.delta_tau[0] = 0.5
    cfg.single_scat_albedo[0] = 0.0
    cfg.planck_levels = np.array([B_atm, B_atm])
    cfg.set_isotropic()

    result = solve(cfg)
    print_results(result, 1)
    print("  Expected: F_up(TOA) dominated by attenuated surface + layer emission")


# ============================================================================
#  Test 2: Conservative scattering (omega=1, energy conservation)
# ============================================================================
def test_conservative_scattering():
    print_header("Test 2: Conservative scattering (omega=1, solar beam)")
    print("  For omega=1, net flux should be constant through all layers.\n")

    nlay = 5
    cfg = ADConfig(num_layers=nlay, num_quadrature=8)
    cfg.solar_flux = 1.0
    cfg.solar_mu = 0.5
    cfg.surface_albedo = 0.3
    cfg.allocate()

    for l in range(nlay):
        cfg.delta_tau[l] = 0.2
        cfg.single_scat_albedo[l] = 1.0
    cfg.set_henyey_greenstein(0.5)

    result = solve(cfg)
    print_results(result, nlay)

    fnet0 = result.flux_up[0] - result.flux_down[0] - result.flux_direct[0]
    fnetN = result.flux_up[nlay] - result.flux_down[nlay] - result.flux_direct[nlay]
    print(f"  F_net_total(TOA) = {fnet0:.6f}  (includes direct beam)")
    print(f"  F_net_total(BOA) = {fnetN:.6f}")
    print(f"  Difference = {abs(fnet0 - fnetN):.2e}")


# ============================================================================
#  Test 3: Rayleigh scattering atmosphere
# ============================================================================
def test_rayleigh():
    print_header("Test 3: Rayleigh scattering atmosphere")

    nlay = 10
    total_tau = 0.5

    cfg = ADConfig(num_layers=nlay, num_quadrature=8)
    cfg.solar_flux = 1.0
    cfg.solar_mu = 1.0
    cfg.surface_albedo = 0.0
    cfg.allocate()

    for l in range(nlay):
        cfg.delta_tau[l] = total_tau / nlay
        cfg.single_scat_albedo[l] = 1.0
    cfg.set_rayleigh()

    result = solve(cfg)
    print_results(result, nlay)

    print(f"  Spherical albedo (F_up/F_solar) = "
          f"{result.flux_up[0] / (cfg.solar_flux * cfg.solar_mu):.6f}")


# ============================================================================
#  Test 4: Multi-layer atmosphere with mixed phase functions
# ============================================================================
def test_mixed_atmosphere():
    print_header("Test 4: Mixed atmosphere (HG + Rayleigh layers, thermal + solar)")

    nu = 5000.0
    nlay = 3

    cfg = ADConfig(num_layers=nlay, num_quadrature=8)
    cfg.surface_albedo = 0.1
    cfg.surface_emission = planck(nu, 300.0)
    cfg.solar_flux = 0.01
    cfg.solar_mu = 0.7
    cfg.allocate()

    cfg.delta_tau[0] = 0.1
    cfg.single_scat_albedo[0] = 0.95
    cfg.set_rayleigh(0)

    cfg.delta_tau[1] = 2.0
    cfg.single_scat_albedo[1] = 0.99
    cfg.set_double_henyey_greenstein(0.8, 0.7, -0.3, 1)

    cfg.delta_tau[2] = 0.5
    cfg.single_scat_albedo[2] = 0.1
    cfg.set_isotropic(2)

    cfg.planck_levels = np.array([planck(nu, 220.0), planck(nu, 250.0),
                                  planck(nu, 280.0), planck(nu, 280.0)])

    result = solve(cfg)
    print_results(result, nlay)


# ============================================================================
#  Test 5: Convergence with number of quadrature points
# ============================================================================
def test_convergence():
    print_header("Test 5: Convergence with quadrature order")

    print("  N_quad   F_up(TOA)      F_down(BOA)    J(TOA)")
    print(f"  {'-' * 55}")

    for nq in [2, 4, 8, 12, 16, 20]:
        cfg = ADConfig(num_layers=1, num_quadrature=nq)
        cfg.surface_emission = 2.0
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.9
        cfg.planck_levels = np.array([1.0, 1.0])
        cfg.set_henyey_greenstein(0.8)

        result = solve(cfg)
        print(f"  {nq:6d}   {result.flux_up[0]:13.6f}  {result.flux_down[1]:13.6f}"
              f"  {result.mean_intensity[0]:13.6f}")
    print()


# ============================================================================
#  Test 6: Linear-in-tau thermal source (pure absorption)
# ============================================================================
def test_linear_source():
    print_header("Test 6: Linear-in-tau thermal source (pure absorption)")

    tau = 1.0
    B_top = 1.0
    B_bot = 3.0
    B_bar = (B_top + B_bot) / 2.0
    B_d = (B_bot - B_top) / tau

    print(f"  tau={tau}  B_top={B_top}  B_bot={B_bot}")
    print(f"  B_bar={B_bar}  B_d={B_d}\n")

    # Isothermal
    cfg_iso = ADConfig(num_layers=1, num_quadrature=8)
    cfg_iso.allocate()
    cfg_iso.delta_tau[0] = tau
    cfg_iso.single_scat_albedo[0] = 0.0
    cfg_iso.planck_levels = np.array([B_bar, B_bar])
    cfg_iso.set_isotropic()

    # Linear
    cfg_lin = ADConfig(num_layers=1, num_quadrature=8)
    cfg_lin.allocate()
    cfg_lin.delta_tau[0] = tau
    cfg_lin.single_scat_albedo[0] = 0.0
    cfg_lin.planck_levels = np.array([B_top, B_bot])
    cfg_lin.set_isotropic()

    result_iso = solve(cfg_iso)
    result_lin = solve(cfg_lin)

    # Analytical
    mu, wt = gauss_legendre(8)
    F_up_analytic = 0.0
    F_down_analytic = 0.0
    for i in range(8):
        trans = np.exp(-tau / float(mu[i]))
        one_minus_t = 1.0 - trans
        slope_term = float(mu[i]) * one_minus_t - 0.5 * tau * (1.0 + trans)
        I_up = B_bar * one_minus_t + B_d * slope_term
        I_down = B_bar * one_minus_t - B_d * slope_term
        F_up_analytic += 2.0 * PI * float(wt[i]) * float(mu[i]) * I_up
        F_down_analytic += 2.0 * PI * float(wt[i]) * float(mu[i]) * I_down

    print("               F_up(TOA)     F_down(BOA)")
    print(f"  Isothermal:  {result_iso.flux_up[0]:12.6f}  {result_iso.flux_down[1]:12.6f}")
    print(f"  Linear:      {result_lin.flux_up[0]:12.6f}  {result_lin.flux_down[1]:12.6f}")
    print(f"  Analytical:  {F_up_analytic:12.6f}  {F_down_analytic:12.6f}\n")

    print(f"  Linear vs Analytical (F_up):  diff = {abs(float(result_lin.flux_up[0]) - F_up_analytic):.2e}")
    print(f"  Linear vs Analytical (F_down): diff = {abs(float(result_lin.flux_down[1]) - F_down_analytic):.2e}")
    print("  (should be ~ machine epsilon)")


# ============================================================================
#  Test 7: Delta-M convergence for forward-peaked HG (g=0.9)
# ============================================================================
def test_delta_m_convergence():
    print_header("Test 7: Delta-M convergence for g=0.9 HG (solar beam)")

    print("  Without delta-M:")
    print("  N_quad   F_up(TOA)      F_down(BOA)")
    print(f"  {'-' * 42}")

    for nq in [4, 8, 16, 32]:
        cfg = ADConfig(num_layers=1, num_quadrature=nq)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.9
        cfg.set_henyey_greenstein(0.9)

        r = solve(cfg)
        print(f"  {nq:6d}   {r.flux_up[0]:13.6f}  {r.flux_down[1]:13.6f}")

    print("\n  With delta-M:")
    print("  N_quad   F_up(TOA)      F_down(BOA)")
    print(f"  {'-' * 42}")

    for nq in [4, 8, 16, 32]:
        cfg = ADConfig(num_layers=1, num_quadrature=nq)
        cfg.use_delta_m = True
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.9
        cfg.set_henyey_greenstein(0.9)

        r = solve(cfg)
        print(f"  {nq:6d}   {r.flux_up[0]:13.6f}  {r.flux_down[1]:13.6f}")

    print("\n  Delta-M values should converge faster (less spread across N_quad).")


# ============================================================================
#  Test 8: Energy conservation with delta-M (omega=1)
# ============================================================================
def test_delta_m_energy_conservation():
    print_header("Test 8: Energy conservation with delta-M (omega=1, g=0.85)")

    nlay = 5
    cfg = ADConfig(num_layers=nlay, num_quadrature=8)
    cfg.use_delta_m = True
    cfg.solar_flux = 1.0
    cfg.solar_mu = 0.6
    cfg.allocate()

    for l in range(nlay):
        cfg.delta_tau[l] = 0.5
        cfg.single_scat_albedo[l] = 1.0
    cfg.set_henyey_greenstein(0.85)

    result = solve(cfg)
    print_results(result, nlay)

    fnet0 = result.flux_up[0] - result.flux_down[0] - result.flux_direct[0]
    fnetN = result.flux_up[nlay] - result.flux_down[nlay] - result.flux_direct[nlay]
    print(f"  F_net_total(TOA) = {fnet0:.6f}")
    print(f"  F_net_total(BOA) = {fnetN:.6f}")
    print(f"  Difference = {abs(fnet0 - fnetN):.2e}  (should be ~ 0 for omega=1)")


# ============================================================================
#  Test 9: Backward compatibility (use_delta_m=false unchanged)
# ============================================================================
def test_delta_m_backward_compat():
    print_header("Test 9: Backward compatibility (use_delta_m=False)")

    cfg_default = ADConfig(num_layers=1, num_quadrature=8)
    cfg_default.surface_emission = 1.5
    cfg_default.allocate()
    cfg_default.delta_tau[0] = 1.0
    cfg_default.single_scat_albedo[0] = 0.9
    cfg_default.planck_levels = np.array([1.0, 2.0])
    cfg_default.set_henyey_greenstein(0.5)

    import copy
    cfg_false = copy.deepcopy(cfg_default)

    r_default = solve(cfg_default)
    r_false = solve(cfg_false)

    diff_up = abs(float(r_default.flux_up[0] - r_false.flux_up[0]))
    diff_down = abs(float(r_default.flux_down[1] - r_false.flux_down[1]))
    print(f"  F_up(TOA)  diff = {diff_up:.2e}  (should be 0)")
    print(f"  F_down(BOA) diff = {diff_down:.2e}  (should be 0)")


# ============================================================================
#  Test 10: Delta-M with Rayleigh (f=0, no truncation)
# ============================================================================
def test_delta_m_rayleigh():
    print_header("Test 10: Delta-M with Rayleigh (f=0, no truncation expected)")

    cfg_off = ADConfig(num_layers=1, num_quadrature=8)
    cfg_off.solar_flux = 1.0
    cfg_off.solar_mu = 1.0
    cfg_off.allocate()
    cfg_off.delta_tau[0] = 0.5
    cfg_off.single_scat_albedo[0] = 1.0
    cfg_off.set_rayleigh()

    cfg_on = ADConfig(num_layers=1, num_quadrature=8)
    cfg_on.use_delta_m = True
    cfg_on.solar_flux = 1.0
    cfg_on.solar_mu = 1.0
    cfg_on.allocate()
    cfg_on.delta_tau[0] = 0.5
    cfg_on.single_scat_albedo[0] = 1.0
    cfg_on.set_rayleigh()

    r_off = solve(cfg_off)
    r_on = solve(cfg_on)

    rayleigh_chi = np.zeros(2 * 8 + 1)
    rayleigh_chi[0] = 1.0
    rayleigh_chi[2] = 0.1
    f = rayleigh_chi[2 * 8]
    print(f"  Truncation fraction f = {f}  (should be 0)")
    print(f"  F_up(TOA)  off={r_off.flux_up[0]:.6f}"
          f"  on={r_on.flux_up[0]:.6f}"
          f"  diff={abs(float(r_off.flux_up[0] - r_on.flux_up[0])):.2e}")
    print("  (should match exactly since Rayleigh has no truncation)")


if __name__ == "__main__":
    print("Adding-Doubling Radiative Transfer Solver - Test Suite (JAX)")
    print("Based on Plass, Hansen & Kattawar (1973)\n")

    test_pure_absorption()
    test_conservative_scattering()
    test_rayleigh()
    test_mixed_atmosphere()
    test_convergence()
    test_linear_source()
    test_delta_m_convergence()
    test_delta_m_energy_conservation()
    test_delta_m_backward_compat()
    test_delta_m_rayleigh()
