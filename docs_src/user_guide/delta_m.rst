Using delta-M scaling
=====================

Delta-M scaling (Wiscombe 1977) makes strongly forward-peaked phase functions
tractable with a modest number of streams.  It is enabled with the
``use_delta_m`` flag and requires one extra Legendre moment per layer.  The
theory is in :doc:`../theory/delta_m`.

.. code-block:: cpp

   adrt::ADConfig cfg(20, 8);
   cfg.use_delta_m = true;        // BEFORE allocate(): provisions 2*M + 1 moments
   cfg.allocate();

   cfg.setHenyeyGreenstein(0.85); // fills all 2*M + 1 moments, incl. chi_{2M}
   for (int l = 0; l < 20; ++l) {
     cfg.delta_tau[l]          = 0.5;
     cfg.single_scat_albedo[l] = 0.99;
   }

   adrt::RTOutput r = adrt::solve(cfg);

When to use it
--------------

* **Do** use delta-M for aerosol/cloud layers with :math:`g \gtrsim 0.7`; it
  can reduce the streams needed for a converged flux by a large factor.
* It has **no effect** on Rayleigh scattering (:math:`f = 0` for :math:`M \geq
  2`).
* The truncation fraction is :math:`f = \chi_{2M}`, so a custom phase function
  must fill the moment at index ``2*num_quadrature`` for scaling to activate.

Consistency of derived quantities
---------------------------------

The direct beam and all cumulative optical depths use the scaled optical depth
:math:`\tau^*`.  The **heating rate / flux divergence**, however, uses the
*unscaled* single-scattering albedo, because the absorption optical depth is
invariant under the scaling.  You therefore get physically correct heating
rates with or without delta-M.
