C++ (CPU) backend
=================

Header: ``adding_doubling.hpp``.  Namespace: ``adrt``.

The C++ backend is the reference implementation.  It uses double precision
throughout, template-specialised fixed-size kernels for
:math:`N = 2, 4, 8, 16, 32` streams (with a dynamic fallback for other :math:`N`),
and is the only backend that exposes the analytic temperature Jacobians.


Solver entry points
-------------------

.. cpp:function:: RTOutput adrt::solve(const ADConfig& config)

   Solve the radiative-transfer problem for one spectral point and return the
   fluxes and mean intensities at every interface.  ``validate()`` is called
   internally.

.. cpp:function:: RTOutput adrt::solve(const ADConfig& config, SolverWorkspace& workspace)

   Same as above, but reuses a caller-owned :cpp:class:`SolverWorkspace` that
   caches Legendre polynomials across layers and across successive calls.  Use
   this when solving many spectral points on the same quadrature grid.  A
   workspace must not be shared between threads.


ADConfig
--------

.. cpp:class:: adrt::ADConfig

   Describes the atmosphere and the solve.  Construct it, set flags, call
   :cpp:func:`allocate`, fill the arrays, then pass it to :cpp:func:`adrt::solve`.
   See :doc:`../user_guide/configuration` for the full field reference.

   .. cpp:function:: ADConfig()
   .. cpp:function:: ADConfig(int nlyr, int nquad = 8)

      Construct with the given number of layers and quadrature streams.

   .. cpp:function:: void allocate()

      Size the per-layer / per-level arrays from the current dimensions and
      flags.  Must be called after setting ``num_layers``, ``num_quadrature``,
      ``use_delta_m``, and ``use_thermal_emission``.  Sets each layer's zeroth
      Legendre moment to 1.

   .. cpp:function:: void validate() const

      Throw ``std::invalid_argument`` if the configuration is inconsistent.

   .. cpp:function:: void setIsotropic(int lc = -1)
   .. cpp:function:: void setRayleigh(int lc = -1)
   .. cpp:function:: void setHenyeyGreenstein(double g, int lc = -1)
   .. cpp:function:: void setDoubleHenyeyGreenstein(double f, double g1, double g2, int lc = -1)

      Fill the Legendre moments of layer ``lc`` (or all layers when
      ``lc = -1``) with the named phase function.  See
      :doc:`../user_guide/phase_functions`.

   Key data members (see :doc:`../user_guide/configuration` for all of them):
   ``num_layers``, ``num_quadrature``, ``use_thermal_emission``,
   ``use_delta_m``, ``use_diffusion_lower_bc``, ``index_from_bottom``,
   ``compute_temperature_jacobian``, ``compute_flux_components``,
   ``surface_albedo``, ``surface_temperature``, ``top_temperature``,
   ``solar_flux``, ``solar_mu``, ``wavenumber_low``, ``wavenumber_high``,
   ``delta_tau``, ``single_scat_albedo``, ``temperature``, ``planck_levels``,
   ``phase_function_moments``.


RTOutput
--------

.. cpp:struct:: adrt::RTOutput

   Results of the solve.  Every array is indexed by interface (``0`` = TOA …
   ``num_layers`` = surface).  See :doc:`../user_guide/outputs`.

   * ``std::vector<double> flux_up, flux_down`` — diffuse hemispheric fluxes.
   * ``std::vector<double> mean_intensity`` — actinic mean intensity (incl. the
     direct beam).
   * ``std::vector<double> flux_divergence`` — net flux divergence (heating
     rate).
   * ``std::vector<double> flux_direct`` — attenuated direct solar beam.
   * ``std::vector<double> net_flux_thermal, net_flux_stellar`` — thermal /
     stellar split of the net upward flux (only if
     ``compute_flux_components``).
   * ``std::vector<std::vector<double>> flux_up_temperature_jac``,
     ``flux_down_temperature_jac``, ``mean_intensity_temperature_jac``,
     ``flux_divergence_temperature_jac`` — analytic temperature Jacobians
     indexed ``[interface][dof]`` (only if ``compute_temperature_jacobian``).


SolverWorkspace
---------------

.. cpp:class:: adrt::SolverWorkspace

   Caller-owned cache of Legendre polynomials, passed to the workspace overload
   of :cpp:func:`adrt::solve`.  Reusing it across calls avoids recomputing the
   polynomials.  Single-thread use only.


Exposed utilities (for testing)
-------------------------------

.. cpp:function:: void adrt::gaussLegendre(int n, std::vector<double>& nodes, std::vector<double>& weights)

   Gauss–Legendre nodes and weights on :math:`[0, 1]`.

.. cpp:function:: void adrt::computePhaseMatricesFromLegendre(const std::vector<double>& chi, const std::vector<double>& mu, const std::vector<double>& weights, DynamicMatrix& Ppp, DynamicMatrix& Ppm)

   Build the azimuthally averaged phase matrices from reduced Legendre
   coefficients (addition theorem + Hansen normalisation).

.. cpp:function:: double adrt::planckFunction(double wnumlo, double wnumhi, double temp)

   Planck function integrated between two wavenumbers [cm⁻¹], in W/m².
