Temperature Jacobians
=====================

For thermal problems the C++ backend can return the analytic derivatives of the
fluxes, mean intensity, and heating rate with respect to every temperature
degree of freedom.  This is invaluable for retrievals and for
radiative-equilibrium temperature correctors, and it costs only a small
fraction of the forward solve because it reuses the forward operators
(:doc:`../theory/jacobian`).

Enabling
--------

.. code-block:: cpp

   adrt::ADConfig cfg(40, 8);
   cfg.use_thermal_emission        = true;
   cfg.compute_temperature_jacobian = true;
   cfg.wavenumber_low  = 500.0;
   cfg.wavenumber_high = 1500.0;
   cfg.allocate();
   // ... fill temperatures, delta_tau, single_scat_albedo, moments ...

   adrt::RTOutput r = adrt::solve(cfg);

The flag is only meaningful when the solve has Planck sources
(``use_thermal_emission`` or ``planck_levels``).


Reading the Jacobians
---------------------

Four arrays are filled, each indexed ``[interface][dof]``:

.. code-block:: cpp

   // d(upward flux at interface k) / d(temperature at level m)
   double dFup = r.flux_up_temperature_jac[k][m];

   // Heating-rate sensitivity at interface k to the surface skin temperature
   int surf_dof = cfg.num_layers + 1;
   double dHdTs = r.flux_divergence_temperature_jac[k][surf_dof];

The ``dof`` axis has ``num_layers + 2`` entries:

* ``dof = 0 .. num_layers`` — derivative w.r.t. the level temperature
  :math:`T_m`;
* ``dof = num_layers + 1`` — derivative w.r.t. the surface skin temperature.

If the solve is driven by ``planck_levels`` instead of ``temperature``, the
arrays hold :math:`\partial/\partial B_m` (the Planck chain rule is skipped).

The available Jacobians are:

* ``flux_up_temperature_jac``
* ``flux_down_temperature_jac``
* ``mean_intensity_temperature_jac``
* ``flux_divergence_temperature_jac`` (the heating-rate Jacobian)


Practical notes
---------------

* The direct solar beam has zero temperature derivative, so the Jacobians are
  purely the diffuse (thermal) response even when a solar beam is present.
* The **heating-rate Jacobian** is diagonally dominant in the mid
  optical-depth range but degenerates deep in the atmosphere, where the field
  is in local thermodynamic equilibrium with the source.  If you build a
  temperature corrector on top of it, read
  :ref:`the depth-limit discussion <heating-depth-limit>` first.
* The feature is available on the templated CPU path (with LU caching) and the
  dynamic fallback.  It is **not** yet ported to CUDA or exposed by the JAX
  ``solve`` (in JAX, use :func:`jax.jacobian` / :func:`jax.grad` on the
  differentiable solver instead — see :doc:`../backends/jax`).
