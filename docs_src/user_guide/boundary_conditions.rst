Boundary conditions
===================

Upper boundary
--------------

The top boundary supplies the downward diffuse intensity at the top of the
atmosphere.  In thermal mode it defaults to :math:`B(T_0)` (the top level
temperature).  Override it with ``top_temperature`` — most commonly
``top_temperature = 0`` for cold space (no downwelling), which matches DisORT.
In non-thermal mode, ``top_emission`` gives the raw isotropic downward
radiance.


Lower boundary: Lambertian surface
----------------------------------

By default the lower boundary is a Lambertian surface characterised by:

* ``surface_albedo`` — the reflectivity in :math:`[0, 1]`;
* ``surface_temperature`` (thermal mode) or ``surface_emission`` (raw) — the
  thermal emission.

If ``surface_temperature`` is left at ``-1`` in thermal mode, the surface emits
at the bottom level temperature ``temperature[num_layers]``.  A distinct
positive ``surface_temperature`` decouples the ground (skin) temperature from
the lowest atmospheric level.

The surface reflects the diffuse field and, if a solar beam is present, also
reflects the attenuated direct beam back upward (:doc:`../theory/boundary_intensity`).


Lower boundary: diffusion approximation
---------------------------------------

For stellar-atmosphere models where the deepest level is an optically thick
interior rather than a physical surface, set ``use_diffusion_lower_bc = true``.
The upward intensity at the lower boundary is then

.. math::

   I^{\uparrow}_{\text{bot},i} = B(T_\text{bot})
     + \mu_i\left.\frac{\dd B}{\dd\tau}\right|_\text{bot},

with the Planck gradient estimated from the bottom two levels.  When this is
active, no surface layer is added, so ``surface_albedo`` and the surface
emission fields are ignored.  The condition requires Planck data
(``use_thermal_emission`` or ``planck_levels``).

.. code-block:: cpp

   adrt::ADConfig cfg(50, 16);
   cfg.use_thermal_emission   = true;
   cfg.use_diffusion_lower_bc = true;
   cfg.wavenumber_low  = 2000.0;
   cfg.wavenumber_high = 3000.0;
   cfg.allocate();
   // ... fill temperatures, delta_tau, single_scat_albedo, moments ...
   adrt::RTOutput r = adrt::solve(cfg);

See :doc:`../theory/boundary_intensity` for the full treatment of the internal
interface intensities.
