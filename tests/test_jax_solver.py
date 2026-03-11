"""Comprehensive test suite for the JAX adding-doubling RT solver.

Tests ported from the C++ test suite (test_ad_solver.cpp) using the same
CDISORT reference values from published benchmark tables.

References:
  VH1 = Van de Hulst (1980), Multiple Light Scattering, Table 12
  VH2 = Van de Hulst (1980), Table 37
  SW  = Sweigart (1970), Table 1
  GS  = Garcia & Siewert (1985), Tables 12-20
  OS  = Ozisik & Shouman, Table 1
"""

import math
import os
import sys

import numpy as np
import pytest

# Add parent directory to path for src_jax import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_jax.config import ADConfig, RTOutput
from src_jax.planck import planck_function
from src_jax.quadrature import gauss_legendre
from src_jax.solver import solve

PI = math.pi


# ============================================================================
#  Helper: Haze-L Garcia-Siewert phase function
# ============================================================================

HAZE_L_MOMENTS = [
    2.41260, 3.23047, 3.37296, 3.23150, 2.89350, 2.49594, 2.11361, 1.74812,
    1.44692, 1.17714, 0.96643, 0.78237, 0.64114, 0.51966, 0.42563, 0.34688,
    0.28351, 0.23317, 0.18963, 0.15788, 0.12739, 0.10762, 0.08597, 0.07381,
    0.05828, 0.05089, 0.03971, 0.03524, 0.02720, 0.02451, 0.01874, 0.01711,
    0.01298, 0.01198, 0.00904, 0.00841, 0.00634, 0.00592, 0.00446, 0.00418,
    0.00316, 0.00296, 0.00225, 0.00210, 0.00160, 0.00150, 0.00115, 0.00107,
    0.00082, 0.00077, 0.00059, 0.00055, 0.00043, 0.00040, 0.00031, 0.00029,
    0.00023, 0.00021, 0.00017, 0.00015, 0.00012, 0.00011, 0.00009, 0.00008,
    0.00006, 0.00006, 0.00005, 0.00004, 0.00004, 0.00003, 0.00003, 0.00002,
    0.00002, 0.00002, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
    0.00001, 0.00001,
]


def set_haze_garcia_siewert(cfg, layer):
    """Set Haze-L Garcia-Siewert phase function on a single layer."""
    nmom = len(cfg.phase_function_moments[layer])
    cfg.phase_function_moments[layer][0] = 1.0
    nmom_limit = min(82, nmom - 1)
    for k in range(1, nmom_limit + 1):
        cfg.phase_function_moments[layer][k] = HAZE_L_MOMENTS[k - 1] / (2 * k + 1)
    for k in range(nmom_limit + 1, nmom):
        cfg.phase_function_moments[layer][k] = 0.0


# ============================================================================
#  Utility Tests
# ============================================================================

class TestUtility:
    def test_gauss_legendre_weight_sum(self):
        """Gauss-Legendre weights on [0,1] should sum to 1."""
        for n in [2, 4, 8, 16, 32]:
            mu, wt = gauss_legendre(n)
            assert len(mu) == n
            assert abs(float(np.sum(wt)) - 1.0) < 1e-14

    def test_gauss_legendre_nodes_in_range(self):
        """All nodes should be in (0, 1)."""
        mu, wt = gauss_legendre(16)
        for i in range(16):
            assert float(mu[i]) > 0.0
            assert float(mu[i]) < 1.0

    def test_planck_function_basic(self):
        """Planck function at 300K, 500-600 cm^-1 should be positive."""
        B = planck_function(500.0, 600.0, 300.0)
        assert B > 0.0

        # Higher temperature -> more emission.
        B_hot = planck_function(500.0, 600.0, 400.0)
        assert B_hot > B

        # Planck at T=0 should be zero.
        assert abs(planck_function(500.0, 600.0, 0.0)) < 1e-30

    def test_planck_stefan_boltzmann(self):
        """Integral over all wavenumbers should approximate sigma*T^4/pi."""
        T = 300.0
        B = planck_function(0.01, 100000.0, T)
        sigma = 5.670374419e-8
        expected = sigma * T**4 / PI
        assert abs(B - expected) < 0.01 * expected


# ============================================================================
#  Test 6b: Pure Absorption — Beer's Law (no scattering)
# ============================================================================

class TestPureAbsorption:
    def test_beers_law(self):
        """1 layer, ssalb=0, beam source. Direct flux follows Beer's law."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = 200.0
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.0
        cfg.set_isotropic()

        r = solve(cfg)

        # At top: flux_direct = mu0 * F0 = 0.5 * 200 = 100
        assert abs(float(r.flux_direct[0]) - 100.0) < 1e-6
        assert abs(float(r.flux_down[0])) < 1e-6
        assert abs(float(r.flux_up[0])) < 1e-6

        # At bottom: flux_direct = 100 * exp(-1/0.5) = 100*exp(-2)
        assert abs(float(r.flux_direct[1]) - 100.0 * math.exp(-2.0)) < 1e-4
        assert abs(float(r.flux_down[1])) < 1e-6
        assert abs(float(r.flux_up[1])) < 1e-6

    def test_lambertian_surface(self):
        """Pure absorption with Lambertian surface (albedo=0.5)."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = 200.0
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.5
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.0
        cfg.set_isotropic()

        r = solve(cfg)

        rfldir_bot = 100.0 * math.exp(-2.0)
        assert abs(float(r.flux_direct[1]) - rfldir_bot) < 1e-4

        flup_bot = 0.5 * rfldir_bot
        assert abs(float(r.flux_up[1]) - flup_bot) < 0.02 * flup_bot

        assert float(r.flux_up[0]) > 0.0
        assert float(r.flux_up[0]) < float(r.flux_up[1])


# ============================================================================
#  Test 1a-1d: Isotropic Scattering (VH1 Table 12)
# ============================================================================

class TestIsotropicScattering:
    def test_1a(self):
        """tau=0.03125, omega=0.2."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = 10.0 * PI
        cfg.solar_mu = 0.1
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 0.03125
        cfg.single_scat_albedo[0] = 0.2
        cfg.set_isotropic()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 3.1416) < tol * 3.1416
        assert abs(float(r.flux_direct[1]) - 2.2984) < tol * 2.2984
        assert abs(float(r.flux_down[0])) < 1e-4
        assert abs(float(r.flux_up[0]) - 7.9945e-02) < tol * 7.9945e-02
        assert abs(float(r.flux_down[1]) - 7.9411e-02) < tol * 7.9411e-02
        assert abs(float(r.flux_up[1])) < 1e-4

    def test_1b(self):
        """tau=0.03125, omega=1.0 (conservative)."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = 10.0 * PI
        cfg.solar_mu = 0.1
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 0.03125
        cfg.single_scat_albedo[0] = 1.0
        cfg.set_isotropic()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 3.1416) < tol * 3.1416
        assert abs(float(r.flux_direct[1]) - 2.2984) < tol * 2.2984
        assert abs(float(r.flux_up[0]) - 0.42292) < tol * 0.42292
        assert abs(float(r.flux_down[1]) - 0.42023) < tol * 0.42023

    def test_1c(self):
        """tau=0.03125, omega=0.99 (nearly conservative)."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = 10.0 * PI
        cfg.solar_mu = 0.1
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 0.03125
        cfg.single_scat_albedo[0] = 0.99
        cfg.set_isotropic()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 3.1416) < tol * 3.1416
        assert abs(float(r.flux_direct[1]) - 2.2984) < tol * 2.2984
        assert float(r.flux_up[0]) > 0.07
        assert float(r.flux_up[0]) < 0.43

    def test_1d(self):
        """tau=32, omega=0.2 (optically thick)."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = 10.0 * PI
        cfg.solar_mu = 0.1
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 32.0
        cfg.single_scat_albedo[0] = 0.2
        cfg.set_isotropic()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 3.1416) < tol * 3.1416
        assert float(r.flux_direct[1]) < 1e-6
        assert not math.isnan(float(r.flux_up[0]))
        assert not math.isnan(float(r.flux_up[1]))


# ============================================================================
#  Test 2a-2d: Rayleigh Scattering (SW Table 1)
# ============================================================================

class TestRayleighScattering:
    def test_2a(self):
        """tau=0.2, omega=0.5."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = PI
        cfg.solar_mu = 0.080442
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 0.2
        cfg.single_scat_albedo[0] = 0.5
        cfg.set_rayleigh()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 2.52716e-01) < tol * 2.52716e-01
        assert abs(float(r.flux_direct[1]) - 2.10311e-02) < tol * 2.10311e-02
        assert abs(float(r.flux_down[0])) < 1e-6
        assert abs(float(r.flux_up[1])) < 1e-6
        assert abs(float(r.flux_up[0]) - 5.35063e-02) < tol * 5.35063e-02
        assert abs(float(r.flux_down[1]) - 4.41794e-02) < tol * 4.41794e-02

    def test_2b(self):
        """tau=0.2, omega=1.0 (conservative Rayleigh)."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = PI
        cfg.solar_mu = 0.080442
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 0.2
        cfg.single_scat_albedo[0] = 1.0
        cfg.set_rayleigh()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 2.52716e-01) < tol * 2.52716e-01
        assert abs(float(r.flux_direct[1]) - 2.10311e-02) < tol * 2.10311e-02
        assert abs(float(r.flux_up[0]) - 1.25561e-01) < tol * 1.25561e-01
        assert abs(float(r.flux_down[1]) - 1.06123e-01) < tol * 1.06123e-01

    def test_2c(self):
        """tau=5.0, omega=0.5 (optically thick Rayleigh)."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = PI
        cfg.solar_mu = 0.080442
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 5.0
        cfg.single_scat_albedo[0] = 0.5
        cfg.set_rayleigh()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 2.52716e-01) < tol * 2.52716e-01
        assert float(r.flux_direct[1]) < 1e-20
        assert abs(float(r.flux_up[0]) - 6.24730e-02) < tol * 6.24730e-02
        assert abs(float(r.flux_down[1]) - 2.51683e-04) < 3e-5

    def test_2d(self):
        """tau=5.0, omega=1.0 (optically thick, conservative Rayleigh)."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = PI
        cfg.solar_mu = 0.080442
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 5.0
        cfg.single_scat_albedo[0] = 1.0
        cfg.set_rayleigh()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 2.52716e-01) < tol * 2.52716e-01
        assert float(r.flux_direct[1]) < 1e-20
        assert abs(float(r.flux_up[0]) - 2.25915e-01) < tol * 2.25915e-01
        assert abs(float(r.flux_down[1]) - 2.68008e-02) < tol * 2.68008e-02


# ============================================================================
#  Test 3a-3b: HG Scattering (VH2 Table 37)
# ============================================================================

class TestHenyeyGreenstein:
    def test_3a(self):
        """tau=1.0, g=0.75, omega=1.0."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = PI
        cfg.solar_mu = 1.0
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 1.0
        cfg.set_henyey_greenstein(0.75)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 3.14159) < tol * 3.14159
        assert abs(float(r.flux_direct[1]) - 1.15573) < tol * 1.15573
        assert abs(float(r.flux_up[0]) - 2.47374e-01) < tol * 2.47374e-01
        assert abs(float(r.flux_down[1]) - 1.73849) < tol * 1.73849
        assert abs(float(r.flux_up[1])) < 1e-6

    def test_3b(self):
        """tau=8.0, g=0.75, omega=1.0 (optically thick HG)."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = PI
        cfg.solar_mu = 1.0
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 8.0
        cfg.single_scat_albedo[0] = 1.0
        cfg.set_henyey_greenstein(0.75)

        r = solve(cfg)

        tol = 5e-2
        assert abs(float(r.flux_direct[0]) - 3.14159) < 2e-2 * 3.14159
        assert abs(float(r.flux_direct[1]) - 1.05389e-03) < 2e-2 * 1.05389e-03
        assert abs(float(r.flux_up[0]) - 1.59096) < tol * 1.59096
        assert abs(float(r.flux_down[1]) - 1.54958) < tol * 1.54958


# ============================================================================
#  Test 4a-4c: Haze-L Garcia-Siewert (GS Tables 12-16)
# ============================================================================

class TestHazeGarciaSiewert:
    def test_4a(self):
        """tau=1.0, omega=1.0, mu0=1.0, nquad=16."""
        cfg = ADConfig(num_layers=1, num_quadrature=16)
        cfg.solar_flux = PI
        cfg.solar_mu = 1.0
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 1.0
        set_haze_garcia_siewert(cfg, 0)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 3.14159) < tol * 3.14159
        assert abs(float(r.flux_direct[1]) - 1.15573) < tol * 1.15573
        assert abs(float(r.flux_up[0]) - 1.73223e-01) < tol * 1.73223e-01
        assert abs(float(r.flux_down[1]) - 1.81264) < tol * 1.81264

    def test_4b(self):
        """tau=1.0, omega=0.9 (absorbing Haze-L)."""
        cfg = ADConfig(num_layers=1, num_quadrature=16)
        cfg.solar_flux = PI
        cfg.solar_mu = 1.0
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.9
        set_haze_garcia_siewert(cfg, 0)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 3.14159) < tol * 3.14159
        assert abs(float(r.flux_direct[1]) - 1.15573) < tol * 1.15573
        assert abs(float(r.flux_up[0]) - 1.23665e-01) < tol * 1.23665e-01
        assert abs(float(r.flux_down[1]) - 1.51554) < tol * 1.51554

    def test_4c(self):
        """tau=1.0, omega=0.9, mu0=0.5 (tilted sun)."""
        cfg = ADConfig(num_layers=1, num_quadrature=16)
        cfg.solar_flux = PI
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.0
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.9
        set_haze_garcia_siewert(cfg, 0)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 1.57080) < tol * 1.57080
        assert abs(float(r.flux_direct[1]) - 2.12584e-01) < tol * 2.12584e-01
        assert abs(float(r.flux_up[0]) - 2.25487e-01) < tol * 2.25487e-01
        assert abs(float(r.flux_down[1]) - 8.03294e-01) < tol * 8.03294e-01


# ============================================================================
#  Multi-Layer Validation Tests
# ============================================================================

class TestMultiLayer:
    def test_two_layer_rayleigh(self):
        """2-layer Rayleigh-like scattering with different properties."""
        cfg = ADConfig(num_layers=2, num_quadrature=8)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.0
        cfg.allocate()

        cfg.delta_tau[0] = 0.1
        cfg.single_scat_albedo[0] = 0.5
        cfg.set_rayleigh(0)

        cfg.delta_tau[1] = 0.2
        cfg.single_scat_albedo[1] = 0.8
        cfg.set_rayleigh(1)

        r = solve(cfg)

        tol = 2e-2
        # Top
        assert abs(float(r.flux_direct[0]) - 5.0e-01) < tol * 5.0e-01
        assert abs(float(r.flux_down[0])) < 1e-6
        assert abs(float(r.flux_up[0]) - 6.593572e-02) < tol * 6.593572e-02
        # Interface
        assert abs(float(r.flux_direct[1]) - 4.093654e-01) < tol * 4.093654e-01
        assert abs(float(r.flux_down[1]) - 2.327019e-02) < tol * 2.327019e-02
        assert abs(float(r.flux_up[1]) - 5.456810e-02) < tol * 5.456810e-02
        # Bottom
        assert abs(float(r.flux_direct[2]) - 2.744058e-01) < tol * 2.744058e-01
        assert abs(float(r.flux_down[2]) - 6.715049e-02) < tol * 6.715049e-02
        assert abs(float(r.flux_up[2])) < 1e-6

    def test_two_layer_hg(self):
        """2-layer HG, equivalent to test 3a split at tau=0.5."""
        cfg = ADConfig(num_layers=2, num_quadrature=8)
        cfg.solar_flux = PI
        cfg.solar_mu = 1.0
        cfg.surface_albedo = 0.0
        cfg.allocate()

        cfg.delta_tau[0] = 0.5
        cfg.single_scat_albedo[0] = 1.0
        cfg.set_henyey_greenstein(0.75, 0)

        cfg.delta_tau[1] = 0.5
        cfg.single_scat_albedo[1] = 1.0
        cfg.set_henyey_greenstein(0.75, 1)

        r = solve(cfg)

        tol = 2e-2
        tol5 = 5e-2
        # Top must match test 3a
        assert abs(float(r.flux_direct[0]) - 3.141593) < tol * 3.141593
        assert abs(float(r.flux_down[0])) < 1e-5
        assert abs(float(r.flux_up[0]) - 2.473744e-01) < tol * 2.473744e-01
        # Middle
        assert abs(float(r.flux_direct[1]) - 1.905472) < tol * 1.905472
        assert abs(float(r.flux_down[1]) - 1.149116) < tol5 * 1.149116
        assert abs(float(r.flux_up[1]) - 1.603704e-01) < tol5 * 1.603704e-01
        # Bottom must match test 3a
        assert abs(float(r.flux_direct[2]) - 1.155727) < tol * 1.155727
        assert abs(float(r.flux_down[2]) - 1.738491) < tol * 1.738491
        assert abs(float(r.flux_up[2])) < 1e-5

    def test_three_layer_isotropic(self):
        """3-layer isotropic with heterogeneous layers and reflective surface."""
        cfg = ADConfig(num_layers=3, num_quadrature=8)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.1
        cfg.allocate()

        cfg.delta_tau[0] = 0.5
        cfg.single_scat_albedo[0] = 0.5
        cfg.set_isotropic(0)
        cfg.delta_tau[1] = 1.0
        cfg.single_scat_albedo[1] = 0.9
        cfg.set_isotropic(1)
        cfg.delta_tau[2] = 0.5
        cfg.single_scat_albedo[2] = 0.5
        cfg.set_isotropic(2)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 5.0e-01) < tol * 5.0e-01
        assert abs(float(r.flux_down[0])) < 1e-6
        assert abs(float(r.flux_up[0]) - 1.133613e-01) < tol * 1.133613e-01

        assert abs(float(r.flux_direct[1]) - 1.839397e-01) < tol * 1.839397e-01
        assert abs(float(r.flux_down[1]) - 6.416506e-02) < tol * 6.416506e-02
        assert abs(float(r.flux_up[1]) - 1.031752e-01) < tol * 1.031752e-01

        assert abs(float(r.flux_direct[2]) - 2.489353e-02) < tol * 2.489353e-02
        assert abs(float(r.flux_down[2]) - 8.605320e-02) < tol * 8.605320e-02
        assert abs(float(r.flux_up[2]) - 1.536433e-02) < tol * 1.536433e-02

        assert abs(float(r.flux_direct[3]) - 9.157819e-03) < tol * 9.157819e-03
        assert abs(float(r.flux_down[3]) - 4.944670e-02) < tol * 4.944670e-02
        assert abs(float(r.flux_up[3]) - 5.860452e-03) < tol * 5.860452e-03


# ============================================================================
#  Test 8a-8c: Diffuse Illumination (OS Table 1)
# ============================================================================

class TestDiffuseIllumination:
    def test_8a(self):
        """ssalb=[0.5, 0.3], dtau=[0.25, 0.25]."""
        cfg = ADConfig(num_layers=2, num_quadrature=4)
        cfg.top_emission = 1.0 / PI
        cfg.surface_albedo = 0.0
        cfg.allocate()

        cfg.delta_tau[0] = 0.25
        cfg.single_scat_albedo[0] = 0.5
        cfg.set_isotropic(0)
        cfg.delta_tau[1] = 0.25
        cfg.single_scat_albedo[1] = 0.3
        cfg.set_isotropic(1)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0])) < 1e-8
        assert abs(float(r.flux_down[0]) - 1.0) < tol * 1.0
        assert abs(float(r.flux_up[0]) - 9.29633e-02) < tol * 9.29633e-02

        assert abs(float(r.flux_direct[1])) < 1e-8
        assert abs(float(r.flux_down[1]) - 7.22235e-01) < tol * 7.22235e-01
        assert abs(float(r.flux_up[1]) - 2.78952e-02) < tol * 2.78952e-02

        assert abs(float(r.flux_direct[2])) < 1e-8
        assert abs(float(r.flux_down[2]) - 5.13132e-01) < tol * 5.13132e-01
        assert abs(float(r.flux_up[2])) < 1e-6

    def test_8b(self):
        """ssalb=[0.8, 0.95], dtau=[0.25, 0.25]."""
        cfg = ADConfig(num_layers=2, num_quadrature=4)
        cfg.top_emission = 1.0 / PI
        cfg.surface_albedo = 0.0
        cfg.allocate()

        cfg.delta_tau[0] = 0.25
        cfg.single_scat_albedo[0] = 0.8
        cfg.set_isotropic(0)
        cfg.delta_tau[1] = 0.25
        cfg.single_scat_albedo[1] = 0.95
        cfg.set_isotropic(1)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_down[0]) - 1.0) < tol * 1.0
        assert abs(float(r.flux_up[0]) - 2.25136e-01) < tol * 2.25136e-01

        assert abs(float(r.flux_down[1]) - 7.95332e-01) < tol * 7.95332e-01
        assert abs(float(r.flux_up[1]) - 1.26349e-01) < tol * 1.26349e-01

        assert abs(float(r.flux_down[2]) - 6.50417e-01) < tol * 6.50417e-01
        assert abs(float(r.flux_up[2])) < 1e-6

    def test_8c(self):
        """ssalb=[0.8, 0.95], dtau=[1.0, 2.0] (thicker layers)."""
        cfg = ADConfig(num_layers=2, num_quadrature=4)
        cfg.top_emission = 1.0 / PI
        cfg.surface_albedo = 0.0
        cfg.allocate()

        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.8
        cfg.set_isotropic(0)
        cfg.delta_tau[1] = 2.0
        cfg.single_scat_albedo[1] = 0.95
        cfg.set_isotropic(1)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_down[0]) - 1.0) < tol * 1.0
        assert abs(float(r.flux_up[0]) - 3.78578e-01) < tol * 3.78578e-01

        assert abs(float(r.flux_down[1]) - 4.86157e-01) < tol * 4.86157e-01
        assert abs(float(r.flux_up[1]) - 2.43397e-01) < tol * 2.43397e-01

        assert abs(float(r.flux_down[2]) - 1.59984e-01) < tol * 1.59984e-01
        assert abs(float(r.flux_up[2])) < 1e-6


# ============================================================================
#  Test 9a-9b: 6-Layer Heterogeneous Atmosphere
# ============================================================================

class TestSixLayer:
    def test_9a_isotropic(self):
        """dtau[l] = l+1, ssalb[l] = 0.65 + l*0.05, isotropic."""
        cfg = ADConfig(num_layers=6, num_quadrature=4)
        cfg.top_emission = 1.0 / PI
        cfg.surface_albedo = 0.0
        cfg.allocate()

        for lc in range(6):
            cfg.delta_tau[lc] = float(lc + 1)
            cfg.single_scat_albedo[lc] = 0.65 + lc * 0.05
            cfg.set_isotropic(lc)

        r = solve(cfg)

        tol5 = 5e-2
        assert abs(float(r.flux_down[0]) - 1.0) < 2e-2 * 1.0
        assert abs(float(r.flux_up[0]) - 2.279734e-01) < tol5 * 2.279734e-01
        assert abs(float(r.flux_up[1]) - 8.750978e-02) < tol5 * 8.750978e-02
        assert abs(float(r.flux_up[6])) < 1e-6

    def test_9b_anisotropic(self):
        """Same as 9a but with DGIS anisotropic phase function."""
        pmom_dgis = [
            1.0,
            2.00916 / 3.0,
            1.56339 / 5.0,
            0.67407 / 7.0,
            0.22215 / 9.0,
            0.04725 / 11.0,
            0.00671 / 13.0,
            0.00068 / 15.0,
            0.00005 / 17.0,
        ]

        cfg = ADConfig(num_layers=6, num_quadrature=4)
        cfg.top_emission = 1.0 / PI
        cfg.surface_albedo = 0.0
        cfg.allocate()

        for lc in range(6):
            cfg.delta_tau[lc] = float(lc + 1)
            cfg.single_scat_albedo[lc] = 0.65 + lc * 0.05
            nmom = len(cfg.phase_function_moments[lc])
            for k in range(min(9, nmom)):
                cfg.phase_function_moments[lc][k] = pmom_dgis[k]
            for k in range(9, nmom):
                cfg.phase_function_moments[lc][k] = 0.0

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_down[0]) - 1.0) < tol * 1.0
        assert abs(float(r.flux_up[0]) - 1.000789e-01) < tol * 1.000789e-01
        assert abs(float(r.flux_up[6])) < 1e-6


# ============================================================================
#  Test 11a-11b: Combined Beam + Isotropic + Surface Albedo
# ============================================================================

class TestCombinedSources:
    def test_11a(self):
        """1 layer, beam + isotropic + surface albedo."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.top_emission = 0.5 / PI
        cfg.surface_albedo = 0.5
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.9
        cfg.set_isotropic()

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 5.000000e-01) < tol * 5.0e-01
        assert abs(float(r.flux_down[0]) - 5.000000e-01) < tol * 5.0e-01
        assert abs(float(r.flux_up[0]) - 5.013753e-01) < tol * 5.013753e-01

        assert abs(float(r.flux_direct[1]) - 6.766764e-02) < tol * 6.766764e-02
        assert abs(float(r.flux_down[1]) - 4.723631e-01) < tol * 4.723631e-01
        assert abs(float(r.flux_up[1]) - 2.700154e-01) < tol * 2.700154e-01

    def test_11b_layer_split(self):
        """Same as 11a but split into 3 layers. Top/bottom must match."""
        cfg = ADConfig(num_layers=3, num_quadrature=8)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.top_emission = 0.5 / PI
        cfg.surface_albedo = 0.5
        cfg.allocate()

        cfg.delta_tau[0] = 0.05
        cfg.single_scat_albedo[0] = 0.9
        cfg.set_isotropic(0)
        cfg.delta_tau[1] = 0.45
        cfg.single_scat_albedo[1] = 0.9
        cfg.set_isotropic(1)
        cfg.delta_tau[2] = 0.50
        cfg.single_scat_albedo[2] = 0.9
        cfg.set_isotropic(2)

        r = solve(cfg)

        tol = 2e-2
        # Top must match 11a
        assert abs(float(r.flux_direct[0]) - 5.000000e-01) < tol * 5.0e-01
        assert abs(float(r.flux_down[0]) - 5.000000e-01) < tol * 5.0e-01
        assert abs(float(r.flux_up[0]) - 5.013753e-01) < tol * 5.013753e-01
        # Bottom must match 11a
        assert abs(float(r.flux_direct[3]) - 6.766764e-02) < tol * 6.766764e-02
        assert abs(float(r.flux_down[3]) - 4.723631e-01) < tol * 4.723631e-01
        assert abs(float(r.flux_up[3]) - 2.700154e-01) < tol * 2.700154e-01


# ============================================================================
#  Thermal Emission Tests
# ============================================================================

class TestThermalEmission:
    def test_7a(self):
        """Planck-only emission, no beam. HG g=0.05."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.surface_albedo = 0.0
        cfg.allocate()

        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.1
        cfg.set_henyey_greenstein(0.05)

        cfg.planck_levels = np.array([
            planck_function(300.0, 800.0, 200.0),
            planck_function(300.0, 800.0, 300.0),
        ])
        cfg.top_emission = 0.0
        cfg.surface_emission = 0.0

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_up[0]) - 8.62935618e+01) < tol * 8.62935618e+01
        assert abs(float(r.flux_down[1]) - 1.21203517e+02) < tol * 1.21203517e+02

    def test_7c(self):
        """Planck + beam + isotropic, HG g=0.8, nstr=12."""
        cfg = ADConfig(num_layers=1, num_quadrature=6)
        cfg.solar_flux = 200.0
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.0
        cfg.allocate()

        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.5
        cfg.set_henyey_greenstein(0.8)

        cfg.planck_levels = np.array([
            planck_function(0.01, 50000.0, 300.0),
            planck_function(0.01, 50000.0, 200.0),
        ])
        cfg.top_emission = 100.0 + planck_function(0.01, 50000.0, 100.0)
        cfg.surface_emission = planck_function(0.01, 50000.0, 320.0)

        r = solve(cfg)

        tol = 2e-2
        assert abs(float(r.flux_direct[0]) - 1.00000000e+02) < tol * 1.00000000e+02
        assert abs(float(r.flux_up[0]) - 4.29571753e+02) < tol * 4.29571753e+02

    def test_thermal_pure_absorption(self):
        """3-layer pure absorption, thermal emission."""
        cfg = ADConfig(num_layers=3, num_quadrature=8)
        cfg.use_thermal_emission = True
        cfg.wavenumber_low = 500.0
        cfg.wavenumber_high = 600.0
        cfg.allocate()

        T = [200.0, 230.0, 260.0, 290.0]
        for l in range(3):
            cfg.delta_tau[l] = 0.5
            cfg.single_scat_albedo[l] = 0.0
            cfg.temperature[l] = T[l]
        cfg.temperature[3] = T[3]
        cfg.set_isotropic()

        r = solve(cfg)

        for l in range(4):
            assert float(r.flux_up[l]) > 0.0
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))
        assert float(r.flux_up[3]) > float(r.flux_up[0])

    def test_thermal_scattering(self):
        """4-layer isotropic, omega=1.0, thermal emission."""
        cfg = ADConfig(num_layers=4, num_quadrature=8)
        cfg.use_thermal_emission = True
        cfg.wavenumber_low = 800.0
        cfg.wavenumber_high = 900.0
        cfg.allocate()

        T = [180.0, 210.0, 240.0, 270.0, 300.0]
        for l in range(4):
            cfg.delta_tau[l] = 0.3
            cfg.single_scat_albedo[l] = 1.0
            cfg.temperature[l] = T[l]
        cfg.temperature[4] = T[4]
        cfg.set_isotropic()

        r = solve(cfg)

        for l in range(5):
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))

    def test_thermal_hg_delta_m(self):
        """Thermal + HG g=0.8, delta-M, surface albedo."""
        cfg = ADConfig(num_layers=3, num_quadrature=8)
        cfg.use_thermal_emission = True
        cfg.use_delta_m = True
        cfg.wavenumber_low = 600.0
        cfg.wavenumber_high = 700.0
        cfg.surface_albedo = 0.3
        cfg.allocate()

        T = [220.0, 240.0, 270.0, 300.0]
        for l in range(3):
            cfg.delta_tau[l] = 1.0
            cfg.single_scat_albedo[l] = 0.9
            cfg.temperature[l] = T[l]
        cfg.temperature[3] = T[3]
        cfg.set_henyey_greenstein(0.8)

        r = solve(cfg)

        for l in range(4):
            assert float(r.flux_up[l]) > 0.0
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))


# ============================================================================
#  Energy Conservation Tests
# ============================================================================

class TestEnergyConservation:
    def test_conservative(self):
        """For omega=1, net flux should be constant through all layers."""
        nlay = 5
        cfg = ADConfig(num_layers=nlay, num_quadrature=8)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.0
        cfg.allocate()

        for l in range(nlay):
            cfg.delta_tau[l] = 0.3
            cfg.single_scat_albedo[l] = 1.0
        cfg.set_henyey_greenstein(0.7)

        r = solve(cfg)

        fnet0 = float(r.flux_up[0]) - float(r.flux_down[0]) - float(r.flux_direct[0])
        for l in range(1, nlay + 1):
            fnet = float(r.flux_up[l]) - float(r.flux_down[l]) - float(r.flux_direct[l])
            assert abs(fnet - fnet0) < 1e-5

    def test_delta_m(self):
        """Energy conservation with delta-M enabled (omega=1)."""
        nlay = 5
        cfg = ADConfig(num_layers=nlay, num_quadrature=8)
        cfg.use_delta_m = True
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.6
        cfg.surface_albedo = 0.0
        cfg.allocate()

        for l in range(nlay):
            cfg.delta_tau[l] = 0.5
            cfg.single_scat_albedo[l] = 1.0
        cfg.set_henyey_greenstein(0.85)

        r = solve(cfg)

        fnet0 = float(r.flux_up[0]) - float(r.flux_down[0]) - float(r.flux_direct[0])
        for l in range(1, nlay + 1):
            fnet = float(r.flux_up[l]) - float(r.flux_down[l]) - float(r.flux_direct[l])
            assert abs(fnet - fnet0) < 1e-5


# ============================================================================
#  Delta-M Tests
# ============================================================================

class TestDeltaM:
    def test_rayleigh_no_effect(self):
        """Rayleigh has f=0, so delta-M should have no effect."""
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

        assert abs(float(r_off.flux_up[0]) - float(r_on.flux_up[0])) < 1e-10
        assert abs(float(r_off.flux_down[1]) - float(r_on.flux_down[1])) < 1e-10

    def test_convergence_faster_for_hg(self):
        """With delta-M, convergence w.r.t. quadrature order should be faster."""
        results_off = []
        results_on = []

        for nq in [4, 16]:
            cfg_off = ADConfig(num_layers=1, num_quadrature=nq)
            cfg_off.solar_flux = 1.0
            cfg_off.solar_mu = 0.5
            cfg_off.allocate()
            cfg_off.delta_tau[0] = 1.0
            cfg_off.single_scat_albedo[0] = 0.9
            cfg_off.set_henyey_greenstein(0.9)
            r_off = solve(cfg_off)
            results_off.append(float(r_off.flux_up[0]))

            cfg_on = ADConfig(num_layers=1, num_quadrature=nq)
            cfg_on.use_delta_m = True
            cfg_on.solar_flux = 1.0
            cfg_on.solar_mu = 0.5
            cfg_on.allocate()
            cfg_on.delta_tau[0] = 1.0
            cfg_on.single_scat_albedo[0] = 0.9
            cfg_on.set_henyey_greenstein(0.9)
            r_on = solve(cfg_on)
            results_on.append(float(r_on.flux_up[0]))

        spread_off = abs(results_off[0] - results_off[1])
        spread_on = abs(results_on[0] - results_on[1])
        assert spread_on < spread_off

    def test_backward_compat(self):
        """use_delta_m=false should give identical results to default."""
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.surface_emission = 1.5
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 0.9
        cfg.planck_levels = np.array([1.0, 2.0])
        cfg.set_henyey_greenstein(0.5)

        r1 = solve(cfg)

        cfg2 = ADConfig(num_layers=1, num_quadrature=8)
        cfg2.use_delta_m = False
        cfg2.surface_emission = 1.5
        cfg2.allocate()
        cfg2.delta_tau[0] = 1.0
        cfg2.single_scat_albedo[0] = 0.9
        cfg2.planck_levels = np.array([1.0, 2.0])
        cfg2.set_henyey_greenstein(0.5)

        r2 = solve(cfg2)

        assert abs(float(r1.flux_up[0]) - float(r2.flux_up[0])) < 1e-14
        assert abs(float(r1.flux_down[1]) - float(r2.flux_down[1])) < 1e-14


# ============================================================================
#  Flux Solver Comparison Tests
# ============================================================================

class TestFluxSolver:
    def test_isotropic_beam(self):
        """2 layers, F0=pi, mu0=0.6, albedo=0.1, omega=0.9."""
        cfg = ADConfig(num_layers=2, num_quadrature=8)
        cfg.solar_flux = PI
        cfg.solar_mu = 0.6
        cfg.surface_albedo = 0.1
        cfg.allocate()

        for lc in range(2):
            cfg.delta_tau[lc] = 0.25
            cfg.single_scat_albedo[lc] = 0.9
            cfg.set_isotropic(lc)

        r = solve(cfg)

        assert float(r.flux_direct[0]) > 0.0
        assert float(r.flux_up[0]) > 0.0
        assert float(r.flux_down[2]) > 0.0
        assert float(r.flux_up[2]) > 0.0

    def test_multi_layer(self):
        """10 layers with varying properties."""
        nlyr = 10
        cfg = ADConfig(num_layers=nlyr, num_quadrature=4)
        cfg.solar_flux = PI
        cfg.solar_mu = 0.7
        cfg.surface_albedo = 0.3
        cfg.allocate()

        for lc in range(nlyr):
            cfg.delta_tau[lc] = 0.05 * (lc + 1)
            cfg.single_scat_albedo[lc] = 0.9 - 0.05 * lc
            cfg.set_henyey_greenstein(0.7, lc)

        r = solve(cfg)

        for l in range(nlyr + 1):
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))
            assert not math.isnan(float(r.flux_direct[l]))
            assert not math.isnan(float(r.mean_intensity[l]))

        # Direct beam should monotonically decrease
        for l in range(nlyr):
            assert float(r.flux_direct[l]) > float(r.flux_direct[l + 1])

    def test_optically_thick(self):
        """Very thick layers (tau=10 each), mostly absorbing."""
        cfg = ADConfig(num_layers=5, num_quadrature=4)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.0
        cfg.allocate()

        for lc in range(5):
            cfg.delta_tau[lc] = 10.0
            cfg.single_scat_albedo[lc] = 0.1
            cfg.set_isotropic(lc)

        r = solve(cfg)

        assert float(r.flux_direct[5]) < 1e-20
        for l in range(6):
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))

    def test_reflective_surface(self):
        """Highly reflective surface (albedo=0.8)."""
        cfg = ADConfig(num_layers=2, num_quadrature=4)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 0.5
        cfg.surface_albedo = 0.8
        cfg.allocate()

        for lc in range(2):
            cfg.delta_tau[lc] = 0.1
            cfg.single_scat_albedo[lc] = 0.9
            cfg.set_isotropic(lc)

        r = solve(cfg)

        assert float(r.flux_up[2]) > 0.3 * (float(r.flux_direct[2]) + float(r.flux_down[2]))


# ============================================================================
#  Configuration Validation Tests
# ============================================================================

class TestConfigValidation:
    def test_rejects_invalid_layers(self):
        cfg = ADConfig(num_layers=0, num_quadrature=8)
        cfg.delta_tau = np.array([])
        cfg.single_scat_albedo = np.array([])
        cfg.phase_function_moments = []
        with pytest.raises(ValueError):
            cfg.validate()

    def test_rejects_negative_tau(self):
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.allocate()
        cfg.delta_tau[0] = -0.1
        with pytest.raises(ValueError):
            cfg.validate()

    def test_rejects_bad_albedo(self):
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.allocate()
        cfg.delta_tau[0] = 1.0
        cfg.single_scat_albedo[0] = 1.5
        with pytest.raises(ValueError):
            cfg.validate()

    def test_allocate_sets_correct_sizes(self):
        cfg = ADConfig(num_layers=3, num_quadrature=8)
        cfg.use_thermal_emission = True
        cfg.allocate()

        assert len(cfg.delta_tau) == 3
        assert len(cfg.single_scat_albedo) == 3
        assert len(cfg.phase_function_moments) == 3
        assert len(cfg.temperature) == 4
        assert len(cfg.phase_function_moments[0]) == 16  # 2*nquad

    def test_henyey_greenstein_moments(self):
        cfg = ADConfig(num_layers=1, num_quadrature=4)
        cfg.allocate()
        g = 0.5
        cfg.set_henyey_greenstein(g)

        for k in range(len(cfg.phase_function_moments[0])):
            expected = g ** k
            assert abs(cfg.phase_function_moments[0][k] - expected) < 1e-14

    def test_rayleigh_moments(self):
        cfg = ADConfig(num_layers=1, num_quadrature=4)
        cfg.allocate()
        cfg.set_rayleigh()

        assert abs(cfg.phase_function_moments[0][0] - 1.0) < 1e-14
        assert abs(cfg.phase_function_moments[0][1] - 0.0) < 1e-14
        assert abs(cfg.phase_function_moments[0][2] - 0.1) < 1e-14
        for k in range(3, len(cfg.phase_function_moments[0])):
            assert abs(cfg.phase_function_moments[0][k]) < 1e-14

    def test_double_henyey_greenstein_moments(self):
        cfg = ADConfig(num_layers=1, num_quadrature=4)
        cfg.allocate()
        f, g1, g2 = 0.7, 0.8, -0.3
        cfg.set_double_henyey_greenstein(f, g1, g2)

        for k in range(len(cfg.phase_function_moments[0])):
            expected = f * g1**k + (1.0 - f) * g2**k
            assert abs(cfg.phase_function_moments[0][k] - expected) < 1e-14


# ============================================================================
#  Linear-in-tau thermal source (analytical comparison)
# ============================================================================

class TestLinearSource:
    def test_pure_absorption(self):
        """Compare against analytical solution for pure absorption."""
        tau = 1.0
        B_top = 1.0
        B_bot = 3.0

        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.allocate()
        cfg.delta_tau[0] = tau
        cfg.single_scat_albedo[0] = 0.0
        cfg.planck_levels = np.array([B_top, B_bot])
        cfg.set_isotropic()

        r = solve(cfg)

        B_bar = (B_top + B_bot) / 2.0
        B_d = (B_bot - B_top) / tau

        mu, wt = gauss_legendre(8)
        F_up_exact = 0.0
        F_down_exact = 0.0
        for i in range(8):
            m = float(mu[i])
            w = float(wt[i])
            trans = math.exp(-tau / m)
            one_minus_t = 1.0 - trans
            slope_term = m * one_minus_t - 0.5 * tau * (1.0 + trans)
            I_up = B_bar * one_minus_t + B_d * slope_term
            I_down = B_bar * one_minus_t - B_d * slope_term
            F_up_exact += 2.0 * PI * w * m * I_up
            F_down_exact += 2.0 * PI * w * m * I_down

        assert abs(float(r.flux_up[0]) - F_up_exact) < 1e-6
        assert abs(float(r.flux_down[1]) - F_down_exact) < 1e-6


# ============================================================================
#  Diffusion Lower Boundary Condition Tests
# ============================================================================

class TestDiffusionBC:
    def test_pure_absorption(self):
        """Pure absorption with diffusion lower BC."""
        cfg = ADConfig(num_layers=3, num_quadrature=8)
        cfg.use_thermal_emission = True
        cfg.use_diffusion_lower_bc = True
        cfg.wavenumber_low = 500.0
        cfg.wavenumber_high = 600.0
        cfg.allocate()

        T = [200.0, 230.0, 260.0, 290.0]
        for l in range(3):
            cfg.delta_tau[l] = 0.5
            cfg.single_scat_albedo[l] = 0.0
            cfg.temperature[l] = T[l]
        cfg.temperature[3] = T[3]
        cfg.set_isotropic()

        r = solve(cfg)

        for l in range(4):
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))
            assert float(r.flux_up[l]) > 0.0

        B_bot = planck_function(500.0, 600.0, 290.0)
        B_top_last = planck_function(500.0, 600.0, 260.0)
        dB_dtau = (B_bot - B_top_last) / 0.5

        F_up_bot_expected = PI * B_bot + (2.0 * PI / 3.0) * dB_dtau
        assert abs(float(r.flux_up[3]) - F_up_bot_expected) < 0.01 * F_up_bot_expected

    def test_scattering(self):
        """Scattering atmosphere with diffusion lower BC."""
        cfg = ADConfig(num_layers=4, num_quadrature=8)
        cfg.use_thermal_emission = True
        cfg.use_diffusion_lower_bc = True
        cfg.wavenumber_low = 800.0
        cfg.wavenumber_high = 900.0
        cfg.allocate()

        T = [180.0, 210.0, 240.0, 270.0, 300.0]
        for l in range(4):
            cfg.delta_tau[l] = 0.3
            cfg.single_scat_albedo[l] = 0.9
            cfg.temperature[l] = T[l]
        cfg.temperature[4] = T[4]
        cfg.set_isotropic()

        r = solve(cfg)

        for l in range(5):
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))
        assert float(r.flux_up[4]) > float(r.flux_up[0])

    def test_ignores_surface(self):
        """When diffusion BC is on, surface_albedo should be ignored."""
        results = []
        for albedo in [0.0, 0.5]:
            cfg = ADConfig(num_layers=2, num_quadrature=8)
            cfg.use_thermal_emission = True
            cfg.use_diffusion_lower_bc = True
            cfg.wavenumber_low = 500.0
            cfg.wavenumber_high = 600.0
            cfg.surface_albedo = albedo
            cfg.allocate()

            cfg.delta_tau[0] = 1.0
            cfg.single_scat_albedo[0] = 0.5
            cfg.delta_tau[1] = 1.0
            cfg.single_scat_albedo[1] = 0.5
            cfg.temperature = np.array([200.0, 250.0, 300.0])
            cfg.set_isotropic()

            r = solve(cfg)
            results.append((float(r.flux_up[0]), float(r.flux_up[2])))

        assert abs(results[0][0] - results[1][0]) < 1e-12
        assert abs(results[0][1] - results[1][1]) < 1e-12

    def test_vs_surface_for_thick_atmosphere(self):
        """Thick bottom layer: diffusion BC and surface BC should agree at TOA."""
        wlo, whi = 500.0, 600.0

        cfg1 = ADConfig(num_layers=3, num_quadrature=8)
        cfg1.use_thermal_emission = True
        cfg1.wavenumber_low = wlo
        cfg1.wavenumber_high = whi
        cfg1.allocate()
        cfg1.delta_tau[0] = 0.5
        cfg1.single_scat_albedo[0] = 0.5
        cfg1.delta_tau[1] = 0.5
        cfg1.single_scat_albedo[1] = 0.5
        cfg1.delta_tau[2] = 50.0
        cfg1.single_scat_albedo[2] = 0.5
        cfg1.temperature = np.array([200.0, 230.0, 260.0, 290.0])
        cfg1.set_isotropic()

        r1 = solve(cfg1)

        cfg2 = ADConfig(num_layers=3, num_quadrature=8)
        cfg2.use_thermal_emission = True
        cfg2.use_diffusion_lower_bc = True
        cfg2.wavenumber_low = wlo
        cfg2.wavenumber_high = whi
        cfg2.allocate()
        cfg2.delta_tau[0] = 0.5
        cfg2.single_scat_albedo[0] = 0.5
        cfg2.delta_tau[1] = 0.5
        cfg2.single_scat_albedo[1] = 0.5
        cfg2.delta_tau[2] = 50.0
        cfg2.single_scat_albedo[2] = 0.5
        cfg2.temperature = np.array([200.0, 230.0, 260.0, 290.0])
        cfg2.set_isotropic()

        r2 = solve(cfg2)

        assert abs(float(r1.flux_up[0]) - float(r2.flux_up[0])) < 0.05 * float(r1.flux_up[0])

    def test_with_delta_m(self):
        """Diffusion BC with delta-M scaling."""
        cfg = ADConfig(num_layers=3, num_quadrature=8)
        cfg.use_thermal_emission = True
        cfg.use_diffusion_lower_bc = True
        cfg.use_delta_m = True
        cfg.wavenumber_low = 600.0
        cfg.wavenumber_high = 700.0
        cfg.allocate()

        T = [220.0, 250.0, 280.0, 310.0]
        for l in range(3):
            cfg.delta_tau[l] = 1.0
            cfg.single_scat_albedo[l] = 0.9
            cfg.temperature[l] = T[l]
        cfg.temperature[3] = T[3]
        cfg.set_henyey_greenstein(0.8)

        r = solve(cfg)

        for l in range(4):
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))
            assert float(r.flux_up[l]) > 0.0

    def test_dynamic_path(self):
        """Diffusion BC through the dynamic (nquad=6) path."""
        cfg = ADConfig(num_layers=2, num_quadrature=6)
        cfg.use_thermal_emission = True
        cfg.use_diffusion_lower_bc = True
        cfg.wavenumber_low = 500.0
        cfg.wavenumber_high = 600.0
        cfg.allocate()

        cfg.delta_tau[0] = 0.5
        cfg.single_scat_albedo[0] = 0.3
        cfg.delta_tau[1] = 0.5
        cfg.single_scat_albedo[1] = 0.3
        cfg.temperature = np.array([200.0, 250.0, 300.0])
        cfg.set_isotropic()

        r = solve(cfg)

        for l in range(3):
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))
            assert float(r.flux_up[l]) > 0.0


# ============================================================================
#  Index from Bottom
# ============================================================================

class TestIndexFromBottom:
    def test_reversal(self):
        """Results with index_from_bottom should be the reverse of default."""
        cfg_top = ADConfig(num_layers=3, num_quadrature=8)
        cfg_top.solar_flux = 1.0
        cfg_top.solar_mu = 0.5
        cfg_top.surface_albedo = 0.1
        cfg_top.allocate()

        cfg_top.delta_tau[0] = 0.1
        cfg_top.single_scat_albedo[0] = 0.5
        cfg_top.delta_tau[1] = 0.5
        cfg_top.single_scat_albedo[1] = 0.9
        cfg_top.delta_tau[2] = 1.0
        cfg_top.single_scat_albedo[2] = 0.3
        cfg_top.set_isotropic()

        r_top = solve(cfg_top)

        cfg_bot = ADConfig(num_layers=3, num_quadrature=8)
        cfg_bot.index_from_bottom = True
        cfg_bot.solar_flux = 1.0
        cfg_bot.solar_mu = 0.5
        cfg_bot.surface_albedo = 0.1
        cfg_bot.allocate()

        # Reversed layer order
        cfg_bot.delta_tau[0] = 1.0
        cfg_bot.single_scat_albedo[0] = 0.3
        cfg_bot.delta_tau[1] = 0.5
        cfg_bot.single_scat_albedo[1] = 0.9
        cfg_bot.delta_tau[2] = 0.1
        cfg_bot.single_scat_albedo[2] = 0.5
        cfg_bot.set_isotropic()

        r_bot = solve(cfg_bot)

        for l in range(4):
            assert abs(float(r_top.flux_up[l]) - float(r_bot.flux_up[3 - l])) < 1e-12
            assert abs(float(r_top.flux_down[l]) - float(r_bot.flux_down[3 - l])) < 1e-12
            assert abs(float(r_top.flux_direct[l]) - float(r_bot.flux_direct[3 - l])) < 1e-12
            assert abs(float(r_top.mean_intensity[l]) - float(r_bot.mean_intensity[3 - l])) < 1e-12


# ============================================================================
#  Mixed Atmosphere
# ============================================================================

class TestMixedAtmosphere:
    def test_mixed(self):
        """3-layer with mixed phase functions, thermal + solar sources."""
        cfg = ADConfig(num_layers=3, num_quadrature=8)
        cfg.surface_albedo = 0.1
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

        cfg.planck_levels = np.array([1.0, 2.0, 3.0, 3.0])
        cfg.surface_emission = 4.0

        r = solve(cfg)

        for l in range(4):
            assert not math.isnan(float(r.flux_up[l]))
            assert not math.isnan(float(r.flux_down[l]))
            assert not math.isnan(float(r.flux_direct[l]))
            assert not math.isnan(float(r.mean_intensity[l]))
            assert float(r.mean_intensity[l]) > 0.0

        assert float(r.flux_direct[0]) > float(r.flux_direct[3])
        assert float(r.flux_up[3]) > 0.0


# ============================================================================
#  Quadrature Convergence
# ============================================================================

class TestQuadratureConvergence:
    def test_convergence(self):
        """Increasing quadrature order should converge."""
        f_up = {}
        for nq in [4, 16, 32]:
            cfg = ADConfig(num_layers=1, num_quadrature=nq)
            cfg.surface_emission = 2.0
            cfg.allocate()
            cfg.delta_tau[0] = 1.0
            cfg.single_scat_albedo[0] = 0.9
            cfg.planck_levels = np.array([1.0, 1.0])
            cfg.set_henyey_greenstein(0.8)

            r = solve(cfg)
            f_up[nq] = float(r.flux_up[0])

        diff_4_32 = abs(f_up[4] - f_up[32])
        diff_16_32 = abs(f_up[16] - f_up[32])
        assert diff_16_32 < diff_4_32


# ============================================================================
#  Rayleigh Spherical Albedo
# ============================================================================

class TestRayleighSphericalAlbedo:
    def test_spherical_albedo(self):
        """Conservative Rayleigh, overhead sun, black surface."""
        nlay = 10
        total_tau = 0.5

        cfg = ADConfig(num_layers=nlay, num_quadrature=16)
        cfg.solar_flux = 1.0
        cfg.solar_mu = 1.0
        cfg.surface_albedo = 0.0
        cfg.allocate()

        for l in range(nlay):
            cfg.delta_tau[l] = total_tau / nlay
            cfg.single_scat_albedo[l] = 1.0
        cfg.set_rayleigh()

        r = solve(cfg)

        albedo = float(r.flux_up[0]) / (cfg.solar_flux * cfg.solar_mu)
        assert 0.15 < albedo < 0.25

        # Energy conservation
        fnet0 = float(r.flux_up[0]) - float(r.flux_down[0]) - float(r.flux_direct[0])
        fnetN = float(r.flux_up[nlay]) - float(r.flux_down[nlay]) - float(r.flux_direct[nlay])
        assert abs(fnet0 - fnetN) < 1e-5


# ============================================================================
#  Solver Validation (rejects invalid configs)
# ============================================================================

class TestSolverValidation:
    def test_rejects_invalid_config(self):
        cfg = ADConfig(num_layers=0, num_quadrature=8)
        cfg.delta_tau = np.array([])
        cfg.single_scat_albedo = np.array([])
        cfg.phase_function_moments = []
        with pytest.raises(ValueError):
            solve(cfg)

    def test_rejects_negative_tau(self):
        cfg = ADConfig(num_layers=1, num_quadrature=8)
        cfg.allocate()
        cfg.delta_tau[0] = -1.0
        with pytest.raises(ValueError):
            solve(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
