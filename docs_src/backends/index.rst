Backends & API reference
========================

AddingDoublingRT provides three backends that implement the same algorithm and
produce matching results.  Choose the one that fits your workload:

.. list-table::
   :header-rows: 1
   :widths: 16 20 32 32

   * - Backend
     - Best for
     - Precision
     - Parallelism
   * - :doc:`C++ (CPU) <cpp>`
     - Single spectral points, one profile at a time, temperature Jacobians.
     - double
     - Template-specialised fixed-size kernels (:math:`N = 2, 4, 8, 16, 32`).
   * - :doc:`CUDA (GPU) <cuda>`
     - Whole spectra, retrieval inner loops.
     - double LU, mixed
     - One CUDA thread per wavenumber, batched.
   * - :doc:`JAX (CPU/GPU) <jax>`
     - Differentiable pipelines, research, prototyping.
     - float64
     - Vectorised across wavenumbers; JIT-compiled; autodiff.

All three share the same physics and configuration model (:doc:`../user_guide/index`).
The pages below document each backend's public API and any backend-specific
data-layout or precision considerations.

.. toctree::
   :maxdepth: 2

   cpp
   cuda
   jax
