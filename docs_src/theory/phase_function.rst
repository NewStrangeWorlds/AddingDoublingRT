Phase-function treatment
========================

Legendre moment convention
--------------------------

All phase functions are specified through their Legendre expansion moments
:math:`\chi_\ell`, defined by

.. math::
   :label: legendre_expansion

   p(\cos\Theta) = \sum_{\ell=0}^{L-1} (2\ell+1)\,\chi_\ell\,
     P_\ell(\cos\Theta),

where :math:`P_\ell` are the Legendre polynomials.  The normalisation
condition is :math:`\chi_0 = 1`.  The :math:`(2\ell+1)` weighting is part of
the sum; the stored moments :math:`\chi_\ell` are the **reduced** expansion
coefficients (so, for example, the asymmetry parameter is
:math:`g = \chi_1`).


Supported phase functions
-------------------------

The configuration provides helpers for the common cases:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Phase function
     - Reduced moments
   * - Isotropic
     - :math:`\chi_0 = 1`, :math:`\chi_\ell = 0` for :math:`\ell \geq 1`.
   * - Rayleigh
     - :math:`\chi_0 = 1`, :math:`\chi_1 = 0`, :math:`\chi_2 = 1/10`,
       :math:`\chi_\ell = 0` for :math:`\ell > 2`.  (The familiar
       :math:`\tfrac12` in the :math:`P_2` term comes from
       :math:`5\chi_2 = \tfrac12`.)
   * - Henyey–Greenstein
     - :math:`\chi_\ell = g^\ell`.
   * - Double Henyey–Greenstein
     - :math:`\chi_\ell = f_w\, g_1^\ell + (1-f_w)\, g_2^\ell`.

Arbitrary phase functions can be used by filling the Legendre moment arrays in
the configuration directly.  See :doc:`../user_guide/phase_functions` for the
API.


Azimuthally averaged phase matrices
-----------------------------------

The :math:`m = 0` phase matrices are built from the Legendre addition theorem:

.. math::
   :label: Ppp

   P^{++}(i,j) &= \frac{1}{2\pi}\sum_{\ell=0}^{L-1}
     (2\ell+1)\,\chi_\ell\, P_\ell(\mu_i)\,P_\ell(\mu_j), \\
   P^{+-}(i,j) &= \frac{1}{2\pi}\sum_{\ell=0}^{L-1}
     (2\ell+1)\,(-1)^\ell\,\chi_\ell\, P_\ell(\mu_i)\,P_\ell(\mu_j).

:math:`P^{++}` describes scattering within the same hemisphere (forward) and
:math:`P^{+-}` into the opposite hemisphere (backward); the
:math:`(-1)^\ell` factor implements :math:`P_\ell(-\mu_j) = (-1)^\ell
P_\ell(\mu_j)`.

**Hansen normalisation** is then applied column-wise: each column :math:`j` is
rescaled so that

.. math::
   :label: hansen

   \sum_{i=1}^{M} \bigl[P^{++}(i,j) + P^{+-}(i,j)\bigr]\,w_i
     = \frac{1}{2\pi}.

This guarantees exact conservation of scattered energy on the discrete
quadrature, which is essential for the doubling recursion to conserve flux.
The Legendre polynomials :math:`P_\ell(\mu_i)` are precomputed with the
standard three-term recurrence and cached.


Solar phase vectors
-------------------

For a direct beam at :math:`\mu_0`, the forward- and backward-scattering phase
vectors are computed analogously:

.. math::
   :label: solar_phase

   p^{+}_i &= \frac{1}{2\pi}\sum_{\ell=0}^{L-1}
     (2\ell+1)\,\chi_\ell\, P_\ell(\mu_i)\,P_\ell(\mu_0), \\
   p^{-}_i &= \frac{1}{2\pi}\sum_{\ell=0}^{L-1}
     (2\ell+1)\,(-1)^\ell\,\chi_\ell\, P_\ell(\mu_i)\,P_\ell(\mu_0),

with the normalisation :math:`\sum_i (p^+_i + p^-_i)\,w_i = 1/(2\pi)`.
