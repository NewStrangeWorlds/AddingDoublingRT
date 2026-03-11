"""Azimuthally-averaged phase matrix construction from Legendre coefficients."""

import jax.numpy as jnp

from .quadrature import precompute_legendre_polynomials

PI = jnp.pi


def compute_phase_matrices(chi, mu, weights, Pl=None):
    """Build azimuthally-averaged phase matrices from Legendre coefficients.

    Args:
        chi: Legendre moments, shape (L,).
        mu: Quadrature nodes, shape (N,).
        weights: Quadrature weights, shape (N,).
        Pl: Optional pre-computed Legendre polynomials, shape (L, N).

    Returns:
        (Ppp, Ppm): Phase matrices, each shape (N, N).
    """
    chi = jnp.asarray(chi)
    mu = jnp.asarray(mu)
    weights = jnp.asarray(weights)
    L = chi.shape[0]
    N = mu.shape[0]

    if Pl is None:
        Pl = precompute_legendre_polynomials(L, mu)

    # l_idx: (L,), signs: (-1)^l
    l_idx = jnp.arange(L)
    signs = (-1.0) ** l_idx
    coeffs = (2 * l_idx + 1) * chi  # (L,)

    # Pl[l, i] * Pl[l, j] -> outer product per l
    # terms[l, i, j] = coeffs[l] * Pl[l, i] * Pl[l, j]
    terms = coeffs[:, None, None] * Pl[:, :, None] * Pl[:, None, :]  # (L, N, N)

    Ppp = jnp.sum(terms, axis=0) / (2.0 * PI)
    Ppm = jnp.sum(signs[:, None, None] * terms, axis=0) / (2.0 * PI)

    # Hansen normalization
    col_sums = jnp.sum((Ppp + Ppm) * weights[:, None], axis=0)  # (N,)
    correction = jnp.where(col_sums > 0.0, 1.0 / (2.0 * PI * col_sums), 1.0)

    Ppp = Ppp * correction[None, :]
    Ppm = Ppm * correction[None, :]

    return Ppp, Ppm


def compute_solar_phase_vectors(chi, mu, weights, mu0, Pl=None):
    """Build solar phase vectors from Legendre coefficients.

    Args:
        chi: Legendre moments, shape (L,).
        mu: Quadrature nodes, shape (N,).
        weights: Quadrature weights, shape (N,).
        mu0: Solar cosine zenith angle.
        Pl: Optional pre-computed Legendre polynomials, shape (L, N).

    Returns:
        (p_plus, p_minus): Vectors, each shape (N,).
    """
    chi = jnp.asarray(chi)
    mu = jnp.asarray(mu)
    weights = jnp.asarray(weights)
    L = chi.shape[0]
    N = mu.shape[0]

    if Pl is None:
        Pl = precompute_legendre_polynomials(L, mu)

    # Legendre polynomials at mu0
    Pl_mu0 = jnp.zeros(L)
    Pl_mu0 = Pl_mu0.at[0].set(1.0)
    if L > 1:
        Pl_mu0 = Pl_mu0.at[1].set(mu0)
    for l in range(2, L):
        Pl_mu0 = Pl_mu0.at[l].set(
            ((2 * l - 1) * mu0 * Pl_mu0[l - 1] - (l - 1) * Pl_mu0[l - 2]) / l
        )

    l_idx = jnp.arange(L)
    signs = (-1.0) ** l_idx
    coeffs = (2 * l_idx + 1) * chi  # (L,)

    # terms[l, i] = coeffs[l] * Pl[l, i] * Pl_mu0[l]
    terms = coeffs[:, None] * Pl * Pl_mu0[:, None]  # (L, N)

    p_plus = jnp.sum(terms, axis=0) / (2.0 * PI)
    p_minus = jnp.sum(signs[:, None] * terms, axis=0) / (2.0 * PI)

    # Normalization
    total = jnp.sum((p_plus + p_minus) * weights)
    correction = jnp.where(total > 0.0, 1.0 / (2.0 * PI * total), 1.0)
    p_plus = p_plus * correction
    p_minus = p_minus * correction

    return p_plus, p_minus
