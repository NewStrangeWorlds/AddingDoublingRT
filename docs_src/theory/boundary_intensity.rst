Boundary conditions and intensity reconstruction
================================================

Once the layer operators are known, the radiation field is fixed by the two
external boundary conditions and reconstructed at every internal interface.


Top of atmosphere
-----------------

The downward diffuse intensity at the top of the atmosphere is set to the
user-supplied isotropic emission field:

.. math::

   I^{\downarrow}_{\mathrm{top},i} = B_{\mathrm{top}}, \qquad i = 1, \dots, M.

In thermal mode this is the Planck function at the topmost level,
:math:`B(T_0)`, unless a separate top-boundary temperature is set (for example
``top_temperature = 0`` for cold space, matching DisORT's default of no
downwelling).  In non-thermal mode it is the raw value ``top_emission``.


Lambertian surface
------------------

A Lambertian surface with albedo :math:`\alpha` and thermal emission
:math:`B_\mathrm{surf}` is represented as an extra layer with

.. math::

   R_{ij} = 2\alpha\,\mu_j w_j\, c, \qquad
   s^{+}_i = (1-\alpha)\,B_\mathrm{surf},

where :math:`c = 1/(2\sum_j \mu_j w_j)` is a quadrature-correction factor that
enforces energy conservation.  The solar surface source is

.. math::

   s^{+}_{i,\mathrm{sol}} = \frac{\alpha}{\pi}\,F_\odot\,\mu_0\,
     \ee^{-\tau^*/\mu_0},

with :math:`\tau^*` the total (possibly delta-M scaled) optical depth of the
atmosphere above the surface.


Diffusion lower boundary condition
----------------------------------

For stellar-atmosphere applications where the deepest level is not a physical
surface but an optically thick interior, a diffusion-approximation lower
boundary can be used instead of a Lambertian surface.  The upward intensity at
the lower boundary is then set to

.. math::
   :label: diffusion_bc

   I^{\uparrow}_{\mathrm{bot},i} = B(T_\mathrm{bot})
     + \mu_i \left.\frac{\dd B}{\dd \tau}\right|_\mathrm{bot},

with the Planck gradient estimated from the bottom two levels,

.. math::

   \left.\frac{\dd B}{\dd \tau}\right|_\mathrm{bot}
     = \frac{B_\mathrm{bot} - B_\mathrm{bot-1}}{\Delta\tau_\mathrm{last}}.

When this boundary condition is active, no surface layer is added to the stack.
It is enabled with ``use_diffusion_lower_bc`` and requires either
``use_thermal_emission`` or explicit ``planck_levels``.


Intensity at internal interfaces
--------------------------------

Composites are built from the top down (``rtop``) and from the bottom up
(``rbase``).  At each internal interface, the layer stack above it
(``rtop``) and below it (``rbase``) are matched by solving for the upward and
downward intensities:

.. math::

   \vect{I}^{\uparrow} &= (\mat{I} - \mat{R}_{ab,\mathrm{base}}
     \mat{R}_{ba,\mathrm{top}})^{-1}
     \bigl[\mat{T}_{ba,\mathrm{base}}\vect{I}_\mathrm{bot} +
     \mat{R}_{ab,\mathrm{base}}(\mat{T}_{ab,\mathrm{top}}\vect{I}_\mathrm{top}
     + \vect{s}^{-}_\mathrm{top}) + \vect{s}^{+}_\mathrm{base}\bigr], \\
   \vect{I}^{\downarrow} &= (\mat{I} - \mat{R}_{ba,\mathrm{top}}
     \mat{R}_{ab,\mathrm{base}})^{-1}
     \bigl[\mat{T}_{ab,\mathrm{top}}\vect{I}_\mathrm{top} +
     \mat{R}_{ba,\mathrm{top}}(\mat{T}_{ba,\mathrm{base}}\vect{I}_\mathrm{bot}
     + \vect{s}^{+}_\mathrm{base}) + \vect{s}^{-}_\mathrm{top}\bigr],

where the source vectors include both the thermal and the solar contributions.
The matrices :math:`(\mat{I} - \mat{R}\mat{R})` are temperature-independent, so
their LU factorisations can be cached and reused when computing the temperature
Jacobian (:doc:`jacobian`).


Fluxes and mean intensity
-------------------------

The upward and downward **diffuse** hemispheric fluxes at each interface are

.. math::

   F^{\uparrow} = 2\pi \sum_{i=1}^{M} w_i\,\mu_i\, I^{\uparrow}_i, \qquad
   F^{\downarrow} = 2\pi \sum_{i=1}^{M} w_i\,\mu_i\, I^{\downarrow}_i.

The direct solar beam is tracked separately:

.. math::

   F_\mathrm{direct}(l) = F_\odot\,\mu_0\,\exp\!\Bigl(-\sum_{k=0}^{l-1}
     \tau^*_k / \mu_0\Bigr),

with :math:`\tau^*_k` the (possibly delta-M scaled) optical depth of layer
:math:`k`.  The total net flux is
:math:`F_\mathrm{net} = F^{\uparrow} - F^{\downarrow} - F_\mathrm{direct}`.

The **actinic mean intensity** :math:`J` follows from the half-range
quadrature,

.. math::
   :label: mean_intensity

   J = \frac{1}{2} \sum_{i=1}^{M} w_i\,
     \bigl(I^{\uparrow}_i + I^{\downarrow}_i\bigr) + \frac{F_\mathrm{direct}}{4\pi\mu_0}.

Following the DisORT convention, :math:`J` is the **full** actinic mean
intensity, including the collimated direct beam contribution
:math:`F_\mathrm{direct}/(4\pi\mu_0)`.  The diffuse-only mean intensity is
recovered as ``mean_intensity - flux_direct / (4*pi*solar_mu)``.


Flux divergence (heating rate)
------------------------------

The net flux divergence at each level is

.. math::
   :label: flux_divergence

   \frac{\dd F}{\dd\tau} = 4\pi\,(1-\omega)\,(J - B),

where :math:`\omega` is the **unscaled** single-scattering albedo.  Following
DisORT, each interface uses the albedo of the layer immediately **above** it
(the top-of-atmosphere level uses the topmost layer).  This quantity is the
radiative heating rate per unit optical depth and is the basis of the
heating-rate Jacobian (:doc:`jacobian`).
