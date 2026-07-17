Quick start
===========

This page shows minimal working examples for each backend.  The
:doc:`user_guide/index` explains every configuration option in detail.

All backends share the same mental model:

#. Build a configuration object describing the atmosphere (number of layers,
   number of quadrature streams, optical depths, single-scattering albedos,
   phase-function moments, boundary conditions, and source terms).
#. Call ``solve`` (single spectral point) or ``solve_batch`` (a whole spectrum).
#. Read the fluxes and mean intensities out of the result, indexed by
   interface from the top of the atmosphere (index ``0``) to the surface
   (index ``num_layers``).


C++: a solar problem
--------------------

.. code-block:: cpp

   #include "adding_doubling.hpp"

   // A 5-layer atmosphere with 8-stream quadrature.
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

   adrt::RTOutput result = adrt::solve(cfg);
   // result.flux_up[0]   -> upward flux at the top of the atmosphere
   // result.flux_down[5] -> downward flux at the surface


C++: a thermal problem
----------------------

.. code-block:: cpp

   adrt::ADConfig cfg(10, 8);
   cfg.use_thermal_emission = true;
   cfg.wavenumber_low       = 500.0;    // cm^-1
   cfg.wavenumber_high      = 1500.0;
   cfg.allocate();

   for (int l = 0; l <= 10; ++l)
     cfg.temperature[l] = 250.0 + 10.0 * l;      // level temperatures [K]
   for (int l = 0; l < 10; ++l) {
     cfg.delta_tau[l]          = 0.3;
     cfg.single_scat_albedo[l] = 0.0;            // pure absorption
   }

   adrt::RTOutput result = adrt::solve(cfg);

By default the surface emits at the bottom level temperature and the model top
emits downward at ``temperature[0]``.  Both boundaries can be decoupled,
following the DisORT convention:

.. code-block:: cpp

   cfg.surface_temperature = 320.0;  // skin temperature, distinct from temperature[num_layers]
   cfg.top_temperature     = 0.0;    // cold space: no downwelling at TOA


JAX: single wavenumber
----------------------

.. code-block:: python

   from src_jax import ADConfig, solve

   cfg = ADConfig()
   cfg.num_layers     = 5
   cfg.num_quadrature = 8
   cfg.solar_flux     = 1.0
   cfg.solar_mu       = 0.5
   cfg.surface_albedo = 0.3
   cfg.allocate()

   for l in range(5):
       cfg.delta_tau[l]          = 0.2
       cfg.single_scat_albedo[l] = 0.9
   cfg.set_henyey_greenstein(0.7)

   result = solve(cfg)
   # result.flux_up[0]   -> upward flux at TOA
   # result.flux_down[5] -> downward flux at BOA


JAX: batched across a spectrum
------------------------------

.. code-block:: python

   import numpy as np
   from src_jax import BatchConfig, solve_batch

   bcfg = BatchConfig()
   bcfg.num_wavenumbers = 1000
   bcfg.num_layers      = 50
   bcfg.num_quadrature  = 8
   bcfg.num_moments_max = 16
   bcfg.surface_albedo  = 0.1
   bcfg.solar_flux      = 1.0    # optional solar beam
   bcfg.solar_mu        = 0.5

   delta_tau = np.random.uniform(0.01, 0.5, (1000, 50))
   ssa       = np.full((1000, 50), 0.9)
   pmom      = np.zeros((50, 16))         # shared across wavenumbers
   for l in range(50):
       for m in range(16):
           pmom[l, m] = 0.7 ** m          # Henyey-Greenstein g = 0.7
   planck    = np.zeros((1000, 51))       # zero = no thermal emission

   flux_up, flux_down = solve_batch(bcfg, delta_tau, ssa, pmom, planck)
   # flux_up.shape  -> (1000,)  TOA upward flux per wavenumber
   # flux_down.shape -> (1000,) TOA downward flux per wavenumber


CUDA: batched host convenience call
-----------------------------------

.. code-block:: cpp

   #include "cuda_solver.cuh"

   adrt::cuda::BatchConfig cfg;
   cfg.num_wavenumbers = 4096;
   cfg.num_layers      = 60;
   cfg.num_quadrature  = 8;
   cfg.num_moments_max = 16;
   cfg.surface_albedo  = 0.1;

   // Flat SoA arrays: array[wav * nlay + layer], etc.
   auto res = adrt::cuda::solveBatchHost(
       cfg, delta_tau, single_scat_albedo,
       phase_moments, /*shared=*/true,
       planck_levels);
   // res.flux_up[w], res.flux_down[w], res.flux_direct[w]

See :doc:`backends/index` for the full API of each backend.
