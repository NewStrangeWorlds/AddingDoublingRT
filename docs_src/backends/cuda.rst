CUDA (GPU) backend
==================

Header: ``cuda_solver.cuh``.  Namespace: ``adrt::cuda``.  Enable with
``-DADRT_ENABLE_CUDA=ON`` at configure time.

The CUDA backend is a **batched** solver: it processes an entire spectrum in
parallel with one thread per wavenumber.  It provides template-specialised
kernels (``solveKernel<N>``) for :math:`N = 2, 4, 8, 16` with register-resident
matrix operations, and a cuBLAS-based path for larger :math:`N`.  Linear
systems are factorised in double precision.


Configuration
-------------

.. cpp:struct:: adrt::cuda::BatchConfig

   Scalars shared across all wavenumbers.  Mirrors ``ADConfig`` but with the
   batch dimension ``num_wavenumbers`` and a ``num_moments_max`` (the maximum
   number of Legendre moments across layers).  Fields include ``num_layers``,
   ``num_quadrature`` (2, 4, 8, 16, or 32), ``use_delta_m``,
   ``use_thermal_emission``, ``use_diffusion_lower_bc``, ``surface_albedo``,
   ``surface_temperature``, ``top_temperature``, ``solar_flux``, ``solar_mu``,
   ``wavenumber_low`` / ``wavenumber_high``, and

   .. cpp:member:: double spectrum_scaling

      A constant factor applied to ``flux_up`` only — for example
      :math:`10^{-3}(R_p/R_s)^2` when producing a planet/star flux ratio.


Device data layouts
-------------------

.. cpp:struct:: adrt::cuda::DeviceData

   Caller-owned **device** pointers using a structure-of-arrays layout,
   ``array[wav * stride + layer_or_level]``.  Inputs include ``delta_tau``,
   ``single_scat_albedo``, ``phase_moments`` (with ``phase_moments_shared`` if
   the same moments apply to every wavenumber), and either ``temperature`` (for
   on-device Planck evaluation) or ``planck_levels``.  Outputs are ``flux_up``,
   ``flux_down``, and optional ``flux_direct``.

.. cpp:struct:: adrt::cuda::RawDeviceData

   A **raw-input** interface for retrieval codes such as BeAR.  Takes
   level-major coefficient arrays with BOA at index 0
   (``absorption_coeff``, ``scattering_coeff``, ``altitude``, ``temperature``,
   ``wavenumber``, and an optional ``cloud_optical_depth``).  The solver
   computes optical depths (trapezoidal rule), single-scattering albedos, and
   the single-wavenumber Planck function inline, and reverses the ordering to
   TOA-first internally.


Entry points
------------

.. cpp:function:: void adrt::cuda::solveBatch(const BatchConfig& config, const DeviceData& data, cudaStream_t stream = 0)

   Launch the batched kernel; all data must already be on the GPU.  Call
   ``uploadQuadratureData()`` once beforehand.

.. cpp:function:: HostResult adrt::cuda::solveBatchHost(const BatchConfig& config, const std::vector<float>& delta_tau, const std::vector<float>& single_scat_albedo, const std::vector<float>& phase_moments, bool phase_moments_shared, const std::vector<float>& planck_levels, const std::vector<float>& temperature = {})

   Convenience wrapper for standalone use: allocates device memory, copies the
   host arrays over, runs the kernel, and copies the results back into a
   ``HostResult`` (``flux_up``, ``flux_down``, ``flux_direct``).

.. cpp:function:: void adrt::cuda::solveBatchFromCoefficients(const BatchConfig& config, const RawDeviceData& data, cudaStream_t stream = 0)

   Solve from raw retrieval inputs for :math:`N \leq 8` with **zero**
   intermediate allocation — optical properties and Planck values are computed
   inside the kernel.  Suitable for very large wavenumber counts.

.. cpp:function:: void adrt::cuda::solveBatchFromCoefficients(const BatchConfig& config, const RawDeviceData& data, SolverWorkspaceGPU& workspace, cudaStream_t stream = 0)

   The :math:`N > 8` overload: preprocesses the raw inputs into intermediate
   arrays held by a persistent :cpp:class:`SolverWorkspaceGPU`, then runs the
   cuBLAS-based solver.

.. cpp:struct:: adrt::cuda::SolverWorkspaceGPU

   Persistent GPU workspace for the cuBLAS path (:math:`N > 8`).  Allocate once
   with ``allocate(nwav, nlay)`` and reuse across retrieval iterations; it
   frees its device memory on destruction.


.. note::

   The CUDA backend uses single precision for the stored optical properties and
   double precision for the LU factorisation of the linear systems.  Analytic
   temperature Jacobians are **not** available on this backend.
