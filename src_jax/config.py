"""Configuration and output data structures for the adding-doubling solver."""

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import numpy as np


@dataclass
class RTOutput:
    """Results of the RT calculation.

    All arrays are indexed by interface number:
      index 0         = top of atmosphere (above layer 0)
      index n_layers  = bottom of atmosphere (below last layer / surface)
    """
    flux_up: jnp.ndarray        # Upward diffuse flux
    flux_down: jnp.ndarray      # Downward diffuse flux
    mean_intensity: jnp.ndarray # J = (1/4pi) integral I dOmega
    flux_direct: jnp.ndarray    # Attenuated direct solar beam flux


@dataclass
class ADConfig:
    """Configuration for the adding-doubling RT solver."""

    num_layers: int = 0
    num_quadrature: int = 8

    use_thermal_emission: bool = False
    use_delta_m: bool = False
    use_diffusion_lower_bc: bool = False
    index_from_bottom: bool = False

    surface_albedo: float = 0.0
    surface_emission: float = 0.0
    top_emission: float = 0.0
    solar_flux: float = 0.0
    solar_mu: float = 1.0

    wavenumber_low: float = 0.0
    wavenumber_high: float = 0.0

    delta_tau: Optional[np.ndarray] = None
    single_scat_albedo: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    planck_levels: Optional[np.ndarray] = None
    phase_function_moments: Optional[list] = None

    def allocate(self):
        """Allocate arrays based on current dimensions and flags."""
        self.delta_tau = np.zeros(self.num_layers)
        self.single_scat_albedo = np.zeros(self.num_layers)

        nmom = 2 * self.num_quadrature + 1 if self.use_delta_m else 2 * self.num_quadrature
        self.phase_function_moments = []
        for _ in range(self.num_layers):
            pm = np.zeros(nmom)
            pm[0] = 1.0
            self.phase_function_moments.append(pm)

        if self.use_thermal_emission:
            self.temperature = np.zeros(self.num_layers + 1)

    def validate(self):
        """Validate configuration."""
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.num_quadrature < 2:
            raise ValueError("num_quadrature must be >= 2")
        if len(self.delta_tau) != self.num_layers:
            raise ValueError("delta_tau size mismatch")
        if len(self.single_scat_albedo) != self.num_layers:
            raise ValueError("single_scat_albedo size mismatch")
        if len(self.phase_function_moments) != self.num_layers:
            raise ValueError("phase_function_moments size mismatch")
        for l in range(self.num_layers):
            if self.delta_tau[l] < 0.0:
                raise ValueError(f"delta_tau[{l}] < 0")
            if self.single_scat_albedo[l] < 0.0 or self.single_scat_albedo[l] > 1.0:
                raise ValueError(f"single_scat_albedo[{l}] out of [0,1]")
        if self.surface_albedo < 0.0 or self.surface_albedo > 1.0:
            raise ValueError("surface_albedo out of [0,1]")
        if self.solar_flux > 0.0 and (self.solar_mu <= 0.0 or self.solar_mu > 1.0):
            raise ValueError("solar_mu must be in (0,1] when solar_flux > 0")
        if self.use_thermal_emission:
            if len(self.temperature) != self.num_layers + 1:
                raise ValueError("temperature size must be num_layers+1")
            for l in range(self.num_layers + 1):
                if self.temperature[l] < 0.0:
                    raise ValueError(f"temperature[{l}] < 0")
            if self.wavenumber_low < 0.0 or self.wavenumber_high <= self.wavenumber_low:
                raise ValueError("invalid wavenumber range")

    def set_henyey_greenstein(self, g, layer=-1):
        """Set single Henyey-Greenstein phase function."""
        start = 0 if layer < 0 else layer
        end = self.num_layers if layer < 0 else layer + 1
        for l in range(start, end):
            nmom = len(self.phase_function_moments[l])
            self.phase_function_moments[l] = np.array([g**k for k in range(nmom)])

    def set_double_henyey_greenstein(self, f, g1, g2, layer=-1):
        """Set double Henyey-Greenstein phase function."""
        start = 0 if layer < 0 else layer
        end = self.num_layers if layer < 0 else layer + 1
        for l in range(start, end):
            nmom = len(self.phase_function_moments[l])
            self.phase_function_moments[l] = np.array(
                [f * g1**k + (1.0 - f) * g2**k for k in range(nmom)]
            )

    def set_isotropic(self, layer=-1):
        """Set isotropic scattering."""
        start = 0 if layer < 0 else layer
        end = self.num_layers if layer < 0 else layer + 1
        for l in range(start, end):
            self.phase_function_moments[l][:] = 0.0
            self.phase_function_moments[l][0] = 1.0

    def set_rayleigh(self, layer=-1):
        """Set Rayleigh scattering phase function."""
        start = 0 if layer < 0 else layer
        end = self.num_layers if layer < 0 else layer + 1
        for l in range(start, end):
            self.phase_function_moments[l][:] = 0.0
            self.phase_function_moments[l][0] = 1.0
            if len(self.phase_function_moments[l]) > 2:
                self.phase_function_moments[l][2] = 0.1
