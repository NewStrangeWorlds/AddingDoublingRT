"""Adding-Doubling Radiative Transfer Solver (JAX implementation).

Implements the matrix operator / adding-doubling method of
Plass, Hansen & Kattawar (1973, Appl. Optics 12, 314) for computing
fluxes and mean intensities in scattering, absorbing, and emitting
plane-parallel atmospheres.
"""

import jax
jax.config.update("jax_enable_x64", True)

from .config import ADConfig, RTOutput
from .solver import solve
from .quadrature import gauss_legendre, precompute_legendre_polynomials
from .phase_matrix import compute_phase_matrices, compute_solar_phase_vectors
from .planck import planck_function
from .batch_solver import BatchConfig, solve_batch

__all__ = [
    "ADConfig",
    "RTOutput",
    "solve",
    "gauss_legendre",
    "precompute_legendre_polynomials",
    "compute_phase_matrices",
    "compute_solar_phase_vectors",
    "planck_function",
    "BatchConfig",
    "solve_batch",
]
