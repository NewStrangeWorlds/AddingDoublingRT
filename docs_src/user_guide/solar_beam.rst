Solar / stellar beam
====================

A collimated beam is enabled by setting ``solar_flux > 0`` together with
``solar_mu``, the cosine of the beam's zenith angle (:math:`\mu_0 \in (0, 1]`).
The beam is attenuated exponentially along its slant path and its scattered
light is tracked through the doubling and adding steps as a separate source
(:doc:`../theory/doubling`).

.. code-block:: cpp

   adrt::ADConfig cfg(5, 8);
   cfg.solar_flux     = 1.0;
   cfg.solar_mu       = 0.5;    // cos(solar zenith angle)
   cfg.surface_albedo = 0.3;
   cfg.allocate();

   for (int l = 0; l < 5; ++l) {
     cfg.delta_tau[l]          = 0.2;
     cfg.single_scat_albedo[l] = 0.9;
   }
   cfg.setHenyeyGreenstein(0.7);

   adrt::RTOutput r = adrt::solve(cfg);


Direct beam and fluxes
----------------------

The attenuated direct beam flux is returned separately in
``flux_direct``:

.. math::

   F_\text{direct}(l) = F_\odot\,\mu_0\,
     \exp\!\Bigl(-\sum_{k=0}^{l-1}\tau^*_k/\mu_0\Bigr),

using the (possibly delta-M scaled) layer optical depths :math:`\tau^*_k`.  The
``flux_up`` and ``flux_down`` arrays contain only the **diffuse** field; the
total net flux is ``flux_up - flux_down - flux_direct``.


Combined thermal + solar
------------------------

Thermal emission and a solar beam can be active in the same solve: set
``use_thermal_emission = true`` and ``solar_flux > 0`` together.  The solver
adds the thermal and solar source vectors at every step, so a single call
returns the combined field.  If you need the two contributions separately, set
``compute_flux_components`` to obtain ``net_flux_thermal`` and
``net_flux_stellar`` (:doc:`outputs`).


Mean intensity and the beam
---------------------------

Remember that ``mean_intensity`` includes the direct beam
(:math:`F_\text{direct}/(4\pi\mu_0)`).  Subtract it to get the diffuse-only
mean intensity — see :doc:`outputs`.
