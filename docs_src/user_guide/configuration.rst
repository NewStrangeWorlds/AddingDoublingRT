Configuration
=============

The atmosphere and the solve are described by a single configuration object —
``ADConfig`` in C++ and JAX.  The workflow is always the same:

#. Set the **dimensions** (``num_layers``, ``num_quadrature``) and any
   **flags** that change array sizes (``use_delta_m``, ``use_thermal_emission``).
#. Call ``allocate()`` to size the per-layer / per-level arrays.
#. Fill the arrays and set the boundary/source parameters.
#. Call ``solve(cfg)``.

``allocate()`` fills the zeroth Legendre moment of every layer with
:math:`1` (an isotropic default) and, in thermal mode, sizes the temperature
array to ``num_layers + 1``.  ``solve()`` calls ``validate()`` internally, but
you may call it yourself to fail early.

.. tip::

   Set ``use_delta_m`` and ``use_thermal_emission`` **before** ``allocate()``:
   delta-M provisions one extra Legendre moment per layer (:math:`2M+1`
   instead of :math:`2M`), and thermal mode allocates the temperature array.


Dimensions and flags
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Field
     - Default
     - Meaning
   * - ``num_layers``
     - ``0``
     - Number of atmospheric layers (must be ``> 0``).
   * - ``num_quadrature``
     - ``8``
     - Gauss–Legendre streams per hemisphere :math:`M` (must be ``>= 2``).
       Fast paths exist for 2, 4, 8, 16, 32.
   * - ``use_thermal_emission``
     - ``false``
     - Compute Planck sources internally from temperatures and the wavenumber
       band.
   * - ``use_delta_m``
     - ``false``
     - Enable delta-M scaling (:doc:`delta_m`).
   * - ``use_diffusion_lower_bc``
     - ``false``
     - Use the diffusion-approximation lower boundary instead of a surface
       (:doc:`boundary_conditions`).
   * - ``index_from_bottom``
     - ``false``
     - If ``true``, user arrays (and outputs) are ordered bottom-of-atmosphere
       first.
   * - ``compute_temperature_jacobian``
     - ``false``
     - Compute analytic temperature Jacobians (thermal only; C++;
       :doc:`jacobians`).
   * - ``compute_flux_components``
     - ``false``
     - Split the net flux into additive thermal and stellar parts
       (:doc:`outputs`).


Boundary and source parameters
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Field
     - Default
     - Meaning
   * - ``surface_albedo``
     - ``0.0``
     - Lambertian surface albedo in :math:`[0, 1]`.
   * - ``surface_emission``
     - ``0.0``
     - Raw surface thermal emission (only when ``!use_thermal_emission``).
   * - ``surface_temperature``
     - ``-1.0``
     - Surface skin temperature [K].  If ``>= 0`` (thermal mode), the surface
       emits at this temperature instead of ``temperature[num_layers]``.
   * - ``top_emission``
     - ``0.0``
     - Raw isotropic downwelling diffuse radiation at TOA (only when
       ``!use_thermal_emission``).
   * - ``top_temperature``
     - ``-1.0``
     - Top-boundary temperature [K].  If ``>= 0`` (thermal mode), TOA
       downwelling is :math:`B(T_\text{top})` instead of :math:`B(T_0)`; set
       ``0`` for cold space (DisORT default).
   * - ``solar_flux``
     - ``0.0``
     - Collimated solar/stellar flux at TOA (set ``> 0`` to enable the beam).
   * - ``solar_mu``
     - ``1.0``
     - Cosine of the solar zenith angle; must be in :math:`(0, 1]` when
       ``solar_flux > 0``.
   * - ``wavenumber_low`` / ``wavenumber_high``
     - ``0.0``
     - Band limits [cm⁻¹] over which the Planck function is integrated
       (thermal mode).


Layer and level arrays
----------------------

After ``allocate()`` these have the sizes shown.  "Layer" arrays have one
entry per layer; "level" arrays have one entry per interface
(``num_layers + 1``).

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - Array
     - Size
     - Meaning
   * - ``delta_tau``
     - ``num_layers``
     - Optical depth of each layer (:math:`\geq 0`).
   * - ``single_scat_albedo``
     - ``num_layers``
     - Single-scattering albedo of each layer, in :math:`[0, 1]`.
   * - ``phase_function_moments``
     - ``num_layers`` × ``nmom``
     - Reduced Legendre moments per layer.  ``nmom`` is :math:`2M` (or
       :math:`2M+1` with delta-M).
   * - ``temperature``
     - ``num_layers + 1``
     - Level temperatures [K] (thermal mode).
   * - ``planck_levels``
     - ``num_layers + 1``
     - Pre-computed Planck values at levels (an alternative to
       ``temperature``; skips the internal Planck evaluation).


Validation rules
----------------

``validate()`` throws (C++ ``std::invalid_argument`` / Python ``ValueError``)
when, among other checks:

* ``num_layers <= 0`` or ``num_quadrature < 2``;
* any array size does not match the declared dimensions;
* a ``delta_tau`` is negative or a ``single_scat_albedo`` leaves
  :math:`[0, 1]`;
* ``surface_albedo`` leaves :math:`[0, 1]`;
* ``solar_flux > 0`` but ``solar_mu`` is not in :math:`(0, 1]`;
* ``surface_temperature`` or ``top_temperature`` is set without
  ``use_thermal_emission`` (use the raw ``*_emission`` fields instead);
* ``use_diffusion_lower_bc`` is set without Planck data.


The RTOutput object
-------------------

``solve()`` returns an ``RTOutput`` whose arrays are all indexed by interface
(``0`` = TOA … ``num_layers`` = surface).  See :doc:`outputs` for the full
list, including the mean-intensity convention and the optional Jacobian and
flux-component arrays.
