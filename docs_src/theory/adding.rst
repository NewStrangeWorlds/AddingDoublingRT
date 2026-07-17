Adding: combining layers
========================

When two layers are combined — layer 1 (top, interfaces :math:`a`–:math:`b`)
above layer 2 (bottom, interfaces :math:`b`–:math:`c`) — the composite
properties are obtained by summing the geometric series of reflections in the
gap at interface :math:`b`.

Because a composite of inhomogeneous sub-layers is itself inhomogeneous, all
four directional matrices are required: :math:`\mat{R}_{ac}, \mat{T}_{ac}`
(illumination from above) and :math:`\mat{R}_{ca}, \mat{T}_{ca}` (illumination
from below).


Resolvents
----------

Two resolvent matrices sum the multiple reflections between the facing
surfaces of the two layers:

.. math::

   \mat{D}_1 = (\mat{I} - \mat{R}_{bc}\mat{R}_{ba})^{-1}, \qquad
   \mat{D}_2 = (\mat{I} - \mat{R}_{ba}\mat{R}_{bc})^{-1}.


Composite reflection and transmission
-------------------------------------

.. math::

   \mat{R}_{ac} &= \mat{R}_{ab} + \mat{T}_{ba}\mat{D}_1\mat{R}_{bc}\mat{T}_{ab}, \\
   \mat{R}_{ca} &= \mat{R}_{cb} + \mat{T}_{bc}\mat{D}_2\mat{R}_{ba}\mat{T}_{cb}, \\
   \mat{T}_{ac} &= \mat{T}_{bc}\mat{D}_2\mat{T}_{ab}, \\
   \mat{T}_{ca} &= \mat{T}_{ba}\mat{D}_1\mat{T}_{cb}.


Composite source vectors
------------------------

The same combination law applies to both the thermal and the solar source
vectors:

.. math::
   :label: sadd

   \vect{s}^{+} &= \vect{s}^{+}_1 + \mat{T}_{ba}\mat{D}_1
     (\vect{s}^{+}_2 + \mat{R}_{bc}\vect{s}^{-}_1), \\
   \vect{s}^{-} &= \vect{s}^{-}_2 + \mat{T}_{bc}\mat{D}_2
     (\vect{s}^{-}_1 + \mat{R}_{ba}\vect{s}^{+}_2).

The upward source emerging from the composite is layer 1's own upward source
plus the light that layer 2 emits (and that layer 1 reflects back down and up)
transmitted up through layer 1, and symmetrically for the downward source.
Because these expressions are **linear** in the source vectors with
temperature-independent operators, differentiating them with respect to the
Planck sources gives the analytic temperature Jacobian at essentially no extra
cost (:doc:`jacobian`).


Non-scattering layer optimisation
---------------------------------

When one of the two layers has :math:`\omega = 0`, its reflection matrices
vanish and its transmission is diagonal,

.. math::

   \mat{R} = \mat{0}, \qquad T_{ij} = \ee^{-\tau/\mu_i}\,\delta_{ij}.

Then :math:`\mat{D}_1 = \mat{D}_2 = \mat{I}`, both matrix inversions disappear,
and the adding step collapses to :math:`O(M^2)` diagonal scaling operations.
The solver detects this case automatically and takes the fast path.
