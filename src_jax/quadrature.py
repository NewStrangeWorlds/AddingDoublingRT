"""Gauss-Legendre quadrature and Legendre polynomial utilities."""

import jax.numpy as jnp
import numpy as np

PI = np.pi


def gauss_legendre(n):
    """Compute Gauss-Legendre quadrature nodes and weights on [0, 1].

    Args:
        n: Number of quadrature points.

    Returns:
        (nodes, weights): Arrays of shape (n,) on [0, 1].
    """
    nodes = np.zeros(n)
    weights = np.zeros(n)

    for i in range((n + 1) // 2):
        z = np.cos(PI * (i + 0.75) / (n + 0.5))

        for _ in range(100):
            p0, p1 = 1.0, z
            for k in range(2, n + 1):
                pk = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k
                p0 = p1
                p1 = pk
            pp = n * (z * p1 - p0) / (z * z - 1.0)
            dz = -p1 / pp
            z += dz
            if abs(dz) < 1e-15:
                break

        w = 2.0 / ((1.0 - z * z) * pp * pp)
        j1 = i
        j2 = n - 1 - i
        nodes[j1] = (1.0 - z) / 2.0
        nodes[j2] = (1.0 + z) / 2.0
        weights[j1] = w / 2.0
        weights[j2] = w / 2.0

    return jnp.array(nodes), jnp.array(weights)


def precompute_legendre_polynomials(L, x):
    """Precompute Legendre polynomials P_l(x) for l=0..L-1 at given x values.

    Args:
        L: Number of Legendre orders.
        x: Evaluation points, shape (nx,).

    Returns:
        Array of shape (L, nx) where Pl[l, i] = P_l(x_i).
    """
    x = jnp.asarray(x)
    nx = x.shape[0]
    Pl = jnp.zeros((L, nx))

    Pl = Pl.at[0].set(jnp.ones(nx))

    if L > 1:
        Pl = Pl.at[1].set(x)

    for l in range(2, L):
        Pl = Pl.at[l].set(((2 * l - 1) * x * Pl[l - 1] - (l - 1) * Pl[l - 2]) / l)

    return Pl
