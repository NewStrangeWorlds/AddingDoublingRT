Thermal emission
================

Set ``use_thermal_emission = true`` to make the solver compute Planck sources
internally from level temperatures and a wavenumber band.  The Planck function
is integrated over ``[wavenumber_low, wavenumber_high]`` (in cm⁻¹) at each
level temperature, and varies **linearly in optical depth** across every layer
(Wiscombe 1976), so a coarse vertical grid still captures the source gradient
accurately.

Basic setup
-----------

.. code-block:: cpp

   adrt::ADConfig cfg(10, 8);
   cfg.use_thermal_emission = true;
   cfg.wavenumber_low  = 500.0;    // cm^-1
   cfg.wavenumber_high = 1500.0;
   cfg.allocate();                 // sizes temperature to num_layers + 1

   for (int l = 0; l <= 10; ++l)
     cfg.temperature[l] = 250.0 + 10.0 * l;   // level temperatures [K]
   for (int l = 0; l < 10; ++l) {
     cfg.delta_tau[l]          = 0.3;
     cfg.single_scat_albedo[l] = 0.0;
   }

   adrt::RTOutput r = adrt::solve(cfg);


Boundary temperatures
---------------------

By default the surface emits at the bottom level temperature
(``temperature[num_layers]``) and the top boundary emits downward at the top
level temperature (``temperature[0]``).  Either can be decoupled, following the
DisORT convention:

.. code-block:: cpp

   cfg.surface_temperature = 320.0;  // skin temperature != temperature[num_layers]
   cfg.top_temperature     = 0.0;    // cold space: no downwelling at TOA

Setting ``top_temperature = 0`` reproduces DisORT's default of no downwelling
diffuse radiation at the top of the atmosphere.  Leaving it at ``-1`` keeps
:math:`B(T_0)`.


Supplying Planck values directly
--------------------------------

If you already have band-integrated Planck values (for example from a
line-by-line radiative-transfer driver), you can bypass the internal Planck
evaluation by filling ``planck_levels`` (length ``num_layers + 1``) instead of
``temperature`` and leaving ``use_thermal_emission = false``.  In that mode
``surface_emission`` and ``top_emission`` provide the raw boundary sources.

.. note::

   The linear-in-:math:`\tau` source, the pure-absorption analytic branch, and
   the diffusion lower boundary condition (:doc:`boundary_conditions`) are all
   part of the thermal treatment.  See :doc:`../theory/doubling` and
   :doc:`../theory/boundary_intensity` for the equations.
