Analytic temperature Jacobians
==============================

For thermal (longwave) problems the C++ backend can return the analytic
derivatives of every output field with respect to every temperature degree of
freedom, at a small fraction of the cost of the forward solve.  This is
enabled by ``compute_temperature_jacobian`` and follows the linearisation of
the interaction principle of Spurr & Christi (2006), specialised to the
temperature variable.


What is differentiated
----------------------

Four output fields are differentiated:

.. math::

   \frac{\partial F^{\uparrow}_k}{\partial T_m},\quad
   \frac{\partial F^{\downarrow}_k}{\partial T_m},\quad
   \frac{\partial J_k}{\partial T_m},\quad
   \frac{\partial (\nabla\!\cdot\!F)_k}{\partial T_m},

for the upward flux, downward flux, actinic mean intensity, and flux
divergence (heating rate).  They fill the four ``RTOutput`` arrays
``flux_up_temperature_jac``, ``flux_down_temperature_jac``,
``mean_intensity_temperature_jac``, and ``flux_divergence_temperature_jac``.

Each array is indexed ``[interface][dof]``:

* **interface** :math:`k = 0, \dots, N_{\text{lyr}}` — the output level.
* **dof** :math:`m` — the temperature it is differentiated against:
  ``dof 0 .. num_layers`` are the level temperatures
  :math:`\partial/\partial T_m`; ``dof num_layers+1`` is the independent
  surface skin temperature :math:`\partial/\partial T_s`.

When the solver is driven by ``planck_levels`` instead of temperatures, the
arrays hold :math:`\partial/\partial B_m` (the Planck chain rule is skipped).


Why it is cheap and exact
-------------------------

Temperature enters the adding–doubling equations **only** through the Planck
emission source.  Every reflection and transmission operator is
temperature-independent,

.. math::

   \frac{\partial \mat{R}}{\partial B_m} = \frac{\partial \mat{T}}{\partial B_m}
     = \mat{0}.

The Jacobian is therefore obtained by re-running the *same forward linear
operators* with differentiated right-hand sides.  It is exact to machine
precision within the discrete-ordinate approximation itself.  The direct solar
beam carries no temperature derivative, so the solar source vectors are ignored
throughout the Jacobian pass.

The implementation works in two stages: it first forms derivatives with respect
to the level Planck values :math:`B_m`, then applies a single per-column Planck
factor :math:`\dd B_m / \dd T_m` at the end.


Source seeds
------------

Each layer's thermal source is written compactly as

.. math::

   \vect{s}^{\uparrow}_l = \vect{a}_l\,\bar B_l + \vect{b}_l\,B'_l,
   \qquad
   \vect{s}^{\downarrow}_l = \vect{a}_l\,\bar B_l - \vect{b}_l\,B'_l,

with layer mean :math:`\bar B_l = \tfrac12(B_l + B_{l+1})` and gradient
:math:`B'_l = (B_{l+1} - B_l)/\tau_l`.  The coefficient vectors
:math:`\vect{a}_l, \vect{b}_l` are temperature-independent (they are the
doubling vectors :math:`\vect{y}_l, \vect{z}_l` in the scattering branch, and
the closed-form absorption coefficients otherwise).  Because :math:`\bar B_l`
and :math:`B'_l` are linear in the two bounding level values, defining

.. math::

   \vect{p}_l \equiv \tfrac12\vect{a}_l - \frac{\vect{b}_l}{\tau_l},
   \qquad
   \vect{q}_l \equiv \tfrac12\vect{a}_l + \frac{\vect{b}_l}{\tau_l},

gives the complete layer-level seed as just two :math:`N`-vectors:

.. math::

   \frac{\partial\vect{s}^{\uparrow}_l}{\partial B_l} = \vect{p}_l,\quad
   \frac{\partial\vect{s}^{\uparrow}_l}{\partial B_{l+1}} = \vect{q}_l,\quad
   \frac{\partial\vect{s}^{\downarrow}_l}{\partial B_l} = \vect{q}_l,\quad
   \frac{\partial\vect{s}^{\downarrow}_l}{\partial B_{l+1}} = \vect{p}_l.

Layer :math:`l` seeds only columns :math:`m = l` and :math:`m = l+1`.  The
surface seed is
:math:`\partial\vect{s}^{\uparrow}/\partial B_s = (1-\alpha)\vect{1}`; the
boundary intensities contribute
:math:`\partial\vect{I}^{\downarrow}_{\text{top}}/\partial B_0 = \vect{1}` and,
for the diffusion lower boundary,
:math:`\partial I^{\uparrow}_{\text{bot},i}/\partial B_L = 1 + \mu_i/\tau_{L-1}`
and :math:`\partial I^{\uparrow}_{\text{bot},i}/\partial B_{L-1} =
-\mu_i/\tau_{L-1}`.


Propagation through adding
--------------------------

The source-combination law :math:numref:`sadd` is linear in the sources with
temperature-independent operators, so it applies **verbatim** to the
derivatives:

.. math::

   \frac{\partial\vect{S}^{+}}{\partial B_m}
     &= \frac{\partial\vect{s}^{+}_1}{\partial B_m}
       + \mat{T}_{ba}\mat{D}_1\!\left(
         \frac{\partial\vect{s}^{+}_2}{\partial B_m}
         + \mat{R}_{bc}\frac{\partial\vect{s}^{-}_1}{\partial B_m}\right), \\
   \frac{\partial\vect{S}^{-}}{\partial B_m}
     &= \frac{\partial\vect{s}^{-}_2}{\partial B_m}
       + \mat{T}_{bc}\mat{D}_2\!\left(
         \frac{\partial\vect{s}^{-}_1}{\partial B_m}
         + \mat{R}_{ba}\frac{\partial\vect{s}^{+}_2}{\partial B_m}\right).

All :math:`N_{\text{lyr}}+2` derivative columns are propagated together as an
:math:`N \times (N_{\text{lyr}}+2)` **block**, so each operator application is a
single matrix product against all degrees of freedom at once.  One bottom-up
sweep produces the source derivatives of every ``rbase`` composite and one
top-down sweep every ``rtop`` composite.

At each interface, the intensity solve
:math:`(\mat{I} - \mat{R}\mat{R})\,\vect{I} = \text{rhs}` uses a
temperature-independent matrix, so the forward LU factorisation is cached and
reused to back-substitute :math:`\partial\vect{I}/\partial B_m` against
:math:`\partial(\text{rhs})/\partial B_m`.  The flux and mean-intensity
derivatives are then the same quadrature reductions as the forward pass.


Heating-rate Jacobian
---------------------

Differentiating :math:numref:`flux_divergence` gives

.. math::
   :label: heating_jac

   \frac{\partial(\nabla\!\cdot\!F)_k}{\partial B_m}
     = 4\pi\,(1-\omega_{l(k)})\left(\frac{\partial J_k}{\partial B_m}
       - \delta_{km}\right),

i.e. the mean-intensity Jacobian scaled by
:math:`4\pi(1-\omega_{l(k)})`, with a diagonal **Planck-cooling** correction
:math:`-4\pi(1-\omega_{l(k)})` on the level-:math:`k` column (from
:math:`\partial B_k/\partial B_m = \delta_{km}`).  The surface column carries
no diagonal term because :math:`B_k` is a level quantity.


Planck chain rule
-----------------

Converting from :math:`B`-Jacobians to :math:`T`-Jacobians is a single
per-column scaling,

.. math::

   \frac{\partial(\cdot)}{\partial T_m}
     = \frac{\partial(\cdot)}{\partial B_m}\,\frac{\dd B_m}{\dd T_m},

using the band-integrated Planck temperature derivative

.. math::

   \frac{\dd B}{\dd T} = \frac{4B}{T}
     - \frac{\sigma}{\pi}\frac{15}{\pi^4}\,T^3
       \bigl(v_1 f(v_1) - v_0 f(v_0)\bigr),
   \qquad v = \frac{c_2 \nu}{T},\ f(v) = \frac{v^3}{\ee^v - 1}.

Level columns use :math:`\dd B_m/\dd T_m` at ``temperature[m]``; the surface
column uses :math:`\dd B_s/\dd T_s` at ``surface_temperature``.


.. _heating-depth-limit:

Depth limit of the heating Jacobian
-----------------------------------

The heating Jacobian :math:`\mat{H} = \partial(\nabla\!\cdot\!F)/\partial T`
is built from the **even** (zeroth) angular moment :math:`J` and carries a
local Planck-cooling diagonal, which makes it diagonally dominant in the
mid optical-depth range.  This is in deliberate contrast to the net-flux
Jacobian :math:`\mat{K} = \partial F/\partial T`, built from the **odd**
(first) moment, whose diagonal cancels by symmetry (it is *identically zero*
on a uniform optical-depth grid) and which is therefore dense and long-ranged.
When a temperature corrector needs a well-conditioned operator, it should drive
the collocated heating residual :math:`\mat{H}`, not the flux residual.

However, :math:`\mat{H}` **degenerates at large optical depth**.  Using the
linear-in-:math:`\tau` source, the self-response of the mean intensity to the
local level temperature has the closed form

.. math::

   \frac{\partial J_i}{\partial B_i} = \tfrac12\bigl[G(a) + G(b)\bigr],
   \qquad
   G(c) = 1 - \frac{1}{2c} + \frac{E_3(c)}{c},

where :math:`a` and :math:`b` are the optical thicknesses of the layers just
above and below level :math:`i`, and :math:`E_n` are the exponential integrals.
In the two limits,

* :math:`G \to 0` as :math:`c \to 0` — optically thin: :math:`J` is insensitive
  to the local temperature (the "top collapse");
* :math:`G \to 1` as :math:`c \to \infty` — optically thick: :math:`J` is
  slaved to the local Planck function.

The heating diagonal is therefore

.. math::

   H_{ii} = 4\pi(1-\omega)\,\frac{\partial B_i}{\partial T_i}
     \left(\tfrac12[G(a)+G(b)] - 1\right)
     \;\xrightarrow[\text{deep}]{}\;
     -\frac{2\pi(1-\omega)}{\Delta\tau}\,\frac{\partial B}{\partial T} \to 0.

The Planck-cooling diagonal is cancelled by the self-response
:math:`\partial J_i/\partial B_i \to 1`, and the whole operator collapses as
:math:`1/(2\Delta\tau)`.  The crossover is governed by the **local layer
optical thickness**, not the cumulative optical depth: by
:math:`\Delta\tau \approx 3` the diagonal is roughly 16 % of its ceiling, and
by :math:`\Delta\tau \approx 10` the heating Jacobian is effectively zero.

.. note::

   This is transport physics, not a numerical defect: at large optical depth
   the field is in local thermodynamic equilibrium with the source
   (:math:`J \to B`), so there is no local radiative restoring force on
   temperature.  A radiative-equilibrium temperature corrector must combine the
   heating Jacobian (which "owns the middle of the optical-depth range") with a
   deep, flux-based diffusion treatment and a thin-top Planck inversion.


Cost, caching, and scope
------------------------

* **No forward work is repeated.**  The one-time :math:`O(N_{\text{lyr}} N^3)`
  doubling and adding costs are reused.  The extra work is two source-only
  adding sweeps plus one differentiated output pass, each
  :math:`O(N_{\text{lyr}}^2 N^2)` for the full block — typically a single-digit
  percentage on top of the forward solve.
* **Caching.**  On the templated CPU path the forward build captures the LU
  factorisations of the adding matrices and interface matrices, so the Jacobian
  never re-factorises.
* **Validation.**  The feature is checked against central finite differences to
  :math:`\sim 10^{-3}` relative error and cross-checked against the sibling
  DisORT++ solver to :math:`\sim 10^{-5}`, over the absorption and scattering
  branches, both boundary conditions, both surface modes, and delta-M on/off.
* **Scope.**  Implemented on the templated CPU path (with caching) and the
  dynamic fallback (Eigen-based, no caching).  A CUDA port is deferred.  Only
  temperature (Planck-source) Jacobians are provided; optical-depth, SSA, and
  phase-function Jacobians are out of scope.
