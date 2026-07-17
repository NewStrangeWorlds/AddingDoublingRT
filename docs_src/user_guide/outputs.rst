Outputs
=======

``solve()`` returns an ``RTOutput`` structure.  Every array is indexed by
**interface** number, from the top of the atmosphere (index ``0``) to the
surface (index ``num_layers``).  With ``index_from_bottom = true`` the ordering
is reversed on output as well.


Always populated
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``flux_up``
     - Upward diffuse hemispheric flux (same units as the source :math:`B`).
   * - ``flux_down``
     - Downward diffuse hemispheric flux.
   * - ``mean_intensity``
     - Actinic mean intensity :math:`J = \frac{1}{4\pi}\int I\,\dd\Omega`,
       **including** the direct stellar beam (DisORT convention).
   * - ``flux_divergence``
     - Net flux divergence :math:`\dd F/\dd\tau = 4\pi(1-\omega)(J - B)` at each
       level; each interface uses the layer immediately above it.
   * - ``flux_direct``
     - Attenuated direct solar-beam flux.

The total net flux is ``flux_up - flux_down - flux_direct``.

.. admonition:: Mean-intensity convention

   ``mean_intensity`` is the **full** actinic mean intensity and includes the
   collimated direct beam contribution :math:`F_\text{direct}/(4\pi\mu_0)`.
   To obtain the diffuse-only mean intensity, subtract it back out::

       J_diffuse = mean_intensity - flux_direct / (4 * pi * solar_mu)


Optional: net-flux components
-----------------------------

When ``compute_flux_components`` is set (C++), the net **upward** flux is split
into additive thermal and stellar parts:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``net_flux_thermal``
     - ``flux_up_thermal - flux_down_thermal`` (Planck-source driven).
   * - ``net_flux_stellar``
     - ``(flux_up_stellar - flux_down_stellar) - flux_direct`` (solar driven,
       including the direct beam).

The split is exact: at frozen opacity the diffuse field is linear in the
sources, and the two arrays sum to the total net flux
``flux_up - flux_down - flux_direct``.  These arrays are empty unless
requested.


Optional: temperature Jacobians
-------------------------------

When ``compute_temperature_jacobian`` is set (thermal problems, C++), four
additional arrays are filled, each indexed ``[interface][dof]``:

* ``flux_up_temperature_jac``
* ``flux_down_temperature_jac``
* ``mean_intensity_temperature_jac``
* ``flux_divergence_temperature_jac``

The ``dof`` axis has ``num_layers + 2`` entries: ``0 .. num_layers`` are the
derivatives with respect to the level temperatures, and ``num_layers + 1`` is
the derivative with respect to the surface skin temperature.  When the solve is
driven by ``planck_levels`` rather than temperatures, these hold
:math:`\partial/\partial B` instead.  See :doc:`jacobians` and
:doc:`../theory/jacobian`.  These arrays are empty unless requested.
