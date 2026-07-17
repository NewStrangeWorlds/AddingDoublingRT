Delta-M scaling
===============

For strongly forward-peaked phase functions (large asymmetry parameter
:math:`g`), a modest number of quadrature streams cannot resolve the narrow
forward lobe.  The delta-M method (Wiscombe 1977) truncates the Legendre
expansion at :math:`2M` terms and absorbs the unresolved forward peak into the
direct beam and the optical properties.  It is enabled by the ``use_delta_m``
flag.


Truncation fraction
-------------------

The fraction of scattered energy in the truncated forward peak is

.. math::

   f = \chi_{2M},

the :math:`2M`-th reduced Legendre moment.  Scaling is applied only when
:math:`f` is non-negligible (:math:`f > 10^{-12}`).  This is why, with
delta-M enabled, ``allocate()`` provisions :math:`2M+1` moments per layer
rather than :math:`2M`.


Scaled optical properties
-------------------------

The layer optical depth and single-scattering albedo are replaced by

.. math::
   :label: tau_star

   \tau^* = (1 - \omega f)\,\tau, \qquad
   \omega^* = \frac{\omega(1-f)}{1-\omega f}.

The absorption optical depth is invariant:
:math:`(1-\omega^*)\tau^* = (1-\omega)\tau`.


Truncated phase function
------------------------

The truncated (rescaled) reduced coefficients are

.. math::
   :label: chi_star

   \chi^*_\ell = \frac{\chi_\ell - f}{1 - f}, \qquad \ell = 0, 1, \dots, 2M-1.

Normalisation is preserved (:math:`\chi^*_0 = 1`).  The phase matrices and
solar phase vectors are built from these truncated coefficients using the
formulae of :doc:`phase_function`.


Effect on the solar beam
------------------------

The forward-scattered light is now treated as part of the direct beam, so the
direct-beam attenuation and all cumulative optical depths use the scaled
:math:`\tau^*` throughout the doubling and adding steps.


Applicability
-------------

For Rayleigh scattering :math:`\chi_\ell = 0` for :math:`\ell > 2`, so
:math:`f = 0` for any :math:`M \geq 2` and no scaling is applied.  Delta-M is
most beneficial for Henyey–Greenstein phase functions with
:math:`g \gtrsim 0.7`, where it dramatically reduces the number of streams
required for convergence.

.. important::

   The **heating-rate** and **flux-divergence** quantities use the *unscaled*
   single-scattering albedo :math:`\omega` in
   :math:numref:`flux_divergence`, even when delta-M scaling is active, because
   the absorption is invariant under the scaling.
