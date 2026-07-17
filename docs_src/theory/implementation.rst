Implementation notes
====================

Code structure (C++)
--------------------

The C++ solver is written in C++17 using the Eigen linear-algebra library for
dense matrix operations.  All code lives in the ``adrt`` namespace.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Contents
   * - ``adding_doubling.hpp``
     - Public API: ``ADConfig``, ``RTOutput``, ``solve()``.
   * - ``matrix.hpp``
     - Fixed-size and dynamic matrix wrappers around Eigen.
   * - ``planck.hpp`` / ``.cpp``
     - Band-integrated Planck function and its temperature derivative.
   * - ``quadrature.hpp`` / ``.cpp``
     - Gauss–Legendre quadrature and Legendre polynomial evaluation.
   * - ``phase_matrix.hpp`` / ``.cpp``
     - Phase-matrix and solar-phase-vector construction.
   * - ``layer.hpp``
     - ``LayerMatrices`` data structures (fixed-size and dynamic).
   * - ``doubling.hpp``
     - Doubling algorithm (templated on the stream count).
   * - ``adding.hpp``
     - Adding algorithm (templated).
   * - ``workspace.hpp``
     - Reusable workspace for caching Legendre polynomials.
   * - ``solver.cpp``
     - Solver dispatch, templated ``solveImpl``, dynamic fallback, and the
       temperature-Jacobian routines.


Template dispatch
-----------------

The core algorithms (doubling, adding, intensity reconstruction) are templated
on the number of quadrature streams :math:`M` as a **compile-time** parameter.
This lets Eigen use fixed-size matrices with stack allocation and SIMD
vectorisation.  At run time, :cpp:func:`adrt::solve` dispatches to explicit
template instantiations for :math:`M = 2, 4, 8, 16, 32`.  For any other stream
count a dynamic fallback based on ``Eigen::MatrixXd`` is used automatically —
correct, but without the fixed-size optimisations.


Adaptive doubling count
-----------------------

The number of initial doublings adapts to the single-scattering albedo, so that
weakly scattering layers do not pay for unnecessary sublayers.  The base count
is

.. math::

   d_0(\omega) = \begin{cases}
     4  & \omega < 0.01, \\
     10 & 0.01 \leq \omega < 0.1, \\
     16 & \omega \geq 0.1,
   \end{cases}

and the total number of doublings is
:math:`d = \max(1,\;\lfloor\log_2\tau\rfloor + d_0)`.  The single-scattering
error of the thin-layer initialisation is :math:`O(\delta\tau^2\omega^2)`, so
weaker scattering tolerates a thicker initial sublayer without loss of
accuracy.


Workspace reuse
---------------

The overload :cpp:func:`adrt::solve` that accepts a
:cpp:class:`adrt::SolverWorkspace` caches the Legendre polynomials
:math:`P_\ell(\mu_i)` across layers and across successive ``solve()`` calls.
This is worthwhile when solving many spectral points on the same quadrature
grid.  Each workspace must be used by a single thread only.


Other notes
-----------

* **Delta-M** scaling is off by default (``use_delta_m = false``) for backward
  compatibility.
* When either layer in an adding step has no scattering, the analytic
  simplification of :doc:`adding` avoids the two :math:`O(M^3)` matrix
  inversions.
* The diffusion lower boundary condition is intended for stellar-atmosphere
  models where the deepest level is not a physical surface; it requires Planck
  data (``use_thermal_emission`` or ``planck_levels``).
* The batched CUDA and JAX backends mirror the same algorithm; see
  :doc:`../backends/index` for their specific data layouts and precision
  choices.
