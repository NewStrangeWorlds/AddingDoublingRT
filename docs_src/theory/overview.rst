Overview and notation
=====================

The problem
-----------

We solve the azimuthally averaged radiative transfer equation for a
plane-parallel atmosphere composed of a stack of homogeneous layers.  Each
layer :math:`l` is characterised by

* an optical depth :math:`\tau_l` (``delta_tau[l]``),
* a single-scattering albedo :math:`\omega_l` (``single_scat_albedo[l]``),
* a set of Legendre phase-function moments :math:`\chi_\ell`
  (``phase_function_moments[l]``),

together with, for thermal problems, the Planck emission at the bounding
levels.

The atmosphere is discretised in the polar-angle cosine :math:`\mu` using
Gauss–Legendre quadrature on the half-range :math:`[0, 1]` with :math:`M`
nodes :math:`\mu_i` and weights :math:`w_i` (``num_quadrature``).  A radiation
field is represented by two length-:math:`M` vectors: the upward intensities
:math:`I^{\uparrow}_i \equiv I(+\mu_i)` and the downward intensities
:math:`I^{\downarrow}_i \equiv I(-\mu_i)`.


The reflection / transmission operators
---------------------------------------

Each layer, and every composite of layers, is described by four operators and
a pair of source vectors:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Symbol
     - Meaning
   * - :math:`\mat{R}_{ab}`
     - Reflection for illumination from **above** (top interface :math:`a`).
   * - :math:`\mat{R}_{ba}`
     - Reflection for illumination from **below** (bottom interface :math:`b`).
   * - :math:`\mat{T}_{ab}`
     - Transmission for illumination from above.
   * - :math:`\mat{T}_{ba}`
     - Transmission for illumination from below.
   * - :math:`\vect{s}^{+}`
     - Internal source emerging upward from the top of the (composite) layer.
   * - :math:`\vect{s}^{-}`
     - Internal source emerging downward from the bottom.

For a **single homogeneous** layer the operators are symmetric under a flip of
the illumination direction: :math:`\mat{R}_{ab} = \mat{R}_{ba}` and
:math:`\mat{T}_{ab} = \mat{T}_{ba}`.  This symmetry is broken as soon as two
different layers are combined by adding, because a composite of inhomogeneous
sub-layers is itself inhomogeneous; the code therefore tracks all four
matrices from the outset.


The two-step algorithm
----------------------

The solve proceeds in two stages:

**Doubling** (:doc:`doubling`)
   builds the :math:`\mat{R}`, :math:`\mat{T}` and source vectors of one
   homogeneous layer of arbitrary optical depth by starting from an
   optically thin sublayer and repeatedly *doubling* it.

**Adding** (:doc:`adding`)
   combines two (possibly inhomogeneous) layers into one composite by
   summing the geometric series of inter-reflections between them.

The full atmosphere is assembled by adding all layers together.  To recover
the intensity at *internal* interfaces, the code builds composites both from
the top down (``rtop``) and from the bottom up (``rbase``) and matches them at
each interface (:doc:`boundary_intensity`).


Index conventions
-----------------

Output arrays are indexed by **interface** (level) number:

.. code-block:: text

   index 0            = top of atmosphere (above layer 0)
   index num_layers   = bottom of atmosphere (below the last layer / surface)

Layer :math:`l` (for :math:`l = 0, \dots, N_{\text{lyr}}-1`) is bounded above
by level :math:`l` and below by level :math:`l+1`.  By default the user
supplies arrays ordered top-of-atmosphere first; setting ``index_from_bottom``
lets you supply and receive them bottom-of-atmosphere first (the solver
reverses internally).
