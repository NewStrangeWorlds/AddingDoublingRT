JAX (CPU/GPU) backend
=====================

Package: ``src_jax``.  Importing it enables 64-bit floating point
(``jax_enable_x64``), which the solver relies on.

The JAX backend is a pure-Python implementation.  It offers a
single-wavenumber solver that matches the C++ API and a batched solver
vectorised across a whole spectrum.  Both are JIT-compilable, run on CPU or
GPU, and — because everything is expressed in ``jax.numpy`` — are
**differentiable** through :func:`jax.grad` and :func:`jax.jacobian`.


Single-wavenumber solver
------------------------

.. py:class:: src_jax.ADConfig

   A dataclass mirroring the C++ ``ADConfig`` with ``snake_case`` fields and
   methods.  Fields: ``num_layers``, ``num_quadrature``,
   ``use_thermal_emission``, ``use_delta_m``, ``use_diffusion_lower_bc``,
   ``index_from_bottom``, ``surface_albedo``, ``surface_emission``,
   ``surface_temperature``, ``top_emission``, ``top_temperature``,
   ``solar_flux``, ``solar_mu``, ``wavenumber_low``, ``wavenumber_high``,
   ``delta_tau``, ``single_scat_albedo``, ``temperature``, ``planck_levels``,
   ``phase_function_moments``.

   Methods: ``allocate()``, ``validate()``, ``set_isotropic(layer=-1)``,
   ``set_rayleigh(layer=-1)``, ``set_henyey_greenstein(g, layer=-1)``,
   ``set_double_henyey_greenstein(f, g1, g2, layer=-1)``.

.. py:class:: src_jax.RTOutput

   Dataclass with ``flux_up``, ``flux_down``, ``mean_intensity``,
   ``flux_divergence``, and ``flux_direct`` (all ``jnp.ndarray``, indexed by
   interface).

.. py:function:: src_jax.solve(config)

   Solve one spectral point.  Returns an :py:class:`RTOutput`.  Equivalent to
   the C++ :cpp:func:`adrt::solve`.

.. code-block:: python

   from src_jax import ADConfig, solve

   cfg = ADConfig()
   cfg.num_layers = 5
   cfg.num_quadrature = 8
   cfg.solar_flux = 1.0
   cfg.solar_mu = 0.5
   cfg.surface_albedo = 0.3
   cfg.allocate()
   for l in range(5):
       cfg.delta_tau[l] = 0.2
       cfg.single_scat_albedo[l] = 0.9
   cfg.set_henyey_greenstein(0.7)

   result = solve(cfg)


Batched solver
--------------

.. py:class:: src_jax.BatchConfig

   Scalars shared across wavenumbers: ``num_wavenumbers``, ``num_layers``,
   ``num_quadrature``, ``num_moments_max``, ``surface_albedo``, ``solar_flux``,
   ``solar_mu``.

.. py:function:: src_jax.solve_batch(config, delta_tau, ssa, phase_moments, planck_levels, use_map=False)

   Solve a batch of wavenumbers.  Arguments:

   * ``delta_tau`` — ``(nwav, nlay)`` optical depths.
   * ``ssa`` — ``(nwav, nlay)`` single-scattering albedos.
   * ``phase_moments`` — ``(nlay, nmom)`` Legendre moments, **shared** across
     wavenumbers.
   * ``planck_levels`` — ``(nwav, nlev)`` Planck values at interfaces (zeros
     for no thermal emission).
   * ``use_map`` — if ``True``, process one wavenumber at a time via
     :func:`jax.lax.map` (cache-friendly on CPU); the default batched kernel is
     better on GPU.

   Returns ``(flux_up_toa, flux_down_boa)``, each of shape ``(nwav,)``.  The
   entire solve — all layers, doubling, and adding — compiles into a single XLA
   program.

.. code-block:: python

   import numpy as np
   from src_jax import BatchConfig, solve_batch

   bcfg = BatchConfig()
   bcfg.num_wavenumbers = 1000
   bcfg.num_layers = 50
   bcfg.num_quadrature = 8
   bcfg.num_moments_max = 16
   bcfg.surface_albedo = 0.1

   delta_tau = np.random.uniform(0.01, 0.5, (1000, 50))
   ssa = np.full((1000, 50), 0.9)
   pmom = np.array([[0.7 ** m for m in range(16)] for _ in range(50)])
   planck = np.zeros((1000, 51))

   flux_up, flux_down = solve_batch(bcfg, delta_tau, ssa, pmom, planck)


Differentiation
---------------

Because the batched solver is a pure JAX function of its array inputs, you can
differentiate the emergent fluxes with respect to any input — optical depths,
single-scattering albedos, phase moments, or Planck values — with standard JAX
transforms:

.. code-block:: python

   import jax

   def toa_up(delta_tau):
       fu, _ = solve_batch(bcfg, delta_tau, ssa, pmom, planck)
       return fu.sum()

   grad_tau = jax.grad(toa_up)(delta_tau)   # d(sum flux_up) / d(delta_tau)

This complements the analytic temperature Jacobians of the C++ backend
(:doc:`../user_guide/jacobians`): use the C++ path for fast, exact temperature
derivatives in production, and the JAX path for flexible autodiff with respect
to arbitrary inputs.


Module-level helpers
--------------------

``src_jax`` also re-exports ``gauss_legendre``,
``precompute_legendre_polynomials``, ``compute_phase_matrices``,
``compute_solar_phase_vectors``, and ``planck_function`` for building or
inspecting the intermediate quantities directly.


.. note::

   The JAX ``solve`` performs the per-layer setup (phase matrices, adaptive
   doubling counts) in NumPy/Python and the linear algebra in JAX.  For a
   fully-JIT single program over a spectrum, use ``solve_batch``.  Precision is
   float64 in ``solve`` and float32-interface / float64-LU in ``solve_batch``.
