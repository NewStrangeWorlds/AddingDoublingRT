Doubling: a single homogeneous layer
====================================

A single homogeneous layer of total optical depth :math:`\tau` and
single-scattering albedo :math:`\omega` is subdivided into :math:`2^d`
identical sublayers of optical depth :math:`\delta\tau = \tau / 2^d`, where the
number of doublings :math:`d` is chosen adaptively
(see :doc:`implementation`).  Starting from the properties of one thin
sublayer, :math:`d` doubling steps produce the properties of the whole layer.


Thin-layer initialisation
-------------------------

For the initial sublayer, the azimuthally averaged (:math:`m = 0`) reflection
and transmission matrices are, to first order in :math:`\delta\tau`,

.. math::

   R_0(i,j) &= \delta\tau\, \Gamma^{-}(i,j), \\
   T_0(i,j) &= \delta_{ij} - \delta\tau\, \Gamma^{+}(i,j),

where

.. math::

   \Gamma^{+}(i,j) &= \frac{1}{\mu_i}
     \Bigl[\delta_{ij} - 2\omega\pi\, P^{++}(i,j)\, w_j\Bigr], \\
   \Gamma^{-}(i,j) &= \frac{2\omega\pi}{\mu_i}\, P^{+-}(i,j)\, w_j.

Here :math:`\mu_i` and :math:`w_i` are the Gauss–Legendre nodes and weights,
and :math:`P^{++}`, :math:`P^{+-}` are the azimuthally averaged forward- and
backward-scattering phase matrices (:doc:`phase_function`).


Thermal source: the :math:`\vect{y}` and :math:`\vect{z}` vectors
-----------------------------------------------------------------

Following Wiscombe (1976), the Planck function is allowed to vary **linearly**
with optical depth across the layer:

.. math::

   B(\tau') = \bar{B} + B_d\Bigl(\tau' - \tfrac{\tau}{2}\Bigr),
   \qquad
   \bar{B} = \frac{B_\mathrm{top} + B_\mathrm{bot}}{2},
   \qquad
   B_d = \frac{B_\mathrm{bot} - B_\mathrm{top}}{\tau}.

Two auxiliary vectors are tracked through the doubling process: :math:`\vect{y}`
(mean emissivity) and :math:`\vect{z}` (emissivity slope).  Their thin-sublayer
values are

.. math::
   :label: yz_init

   y_{0,i} = \frac{(1-\omega)\,\delta\tau}{\mu_i}, \qquad z_{0,i} = 0.


Solar source initialisation
---------------------------

For an incident collimated solar flux :math:`F_\odot` at cosine zenith angle
:math:`\mu_0`, the solar source vectors of the thin sublayer are

.. math::

   s^{+}_{0,\mathrm{sol},i} &= \frac{\omega\,\delta\tau}{\mu_i}\,
     p^{-}_i\, F_\mathrm{top}, \\
   s^{-}_{0,\mathrm{sol},i} &= \frac{\omega\,\delta\tau}{\mu_i}\,
     p^{+}_i\, F_\mathrm{top},

where :math:`F_\mathrm{top} = F_\odot \exp(-\tau_\mathrm{cum}/\mu_0)` is the
attenuated direct beam at the top of the current layer and :math:`p^{\pm}_i`
are the solar phase vectors for scattering from :math:`\mu_0` into stream
:math:`\mu_i` (:doc:`phase_function`).


The doubling loop
-----------------

At each step :math:`k = 0, \dots, d-1` the reflection and transmission
matrices are combined with themselves:

.. math::

   \mat{\Gamma}_k &= (\mat{I} - \mat{R}_k^2)^{-1}, \\
   \mat{R}_{k+1}  &= \mat{R}_k + \mat{T}_k\mat{\Gamma}_k\mat{R}_k\mat{T}_k, \\
   \mat{T}_{k+1}  &= \mat{T}_k\mat{\Gamma}_k\mat{T}_k.

The resolvent :math:`\mat{\Gamma}_k = (\mat{I}-\mat{R}_k^2)^{-1}` sums the
infinite series of internal reflections between the two identical halves.

**Thermal source doubling.**  With :math:`g_k = 2^{k-1}\delta\tau`
(:math:`g_0 = \delta\tau/2`):

.. math::

   \vect{z}_{k+1} &= (\mat{T}_k\mat{\Gamma}_k - \mat{T}_k\mat{\Gamma}_k\mat{R}_k)
     (\vect{z}_k + g_k\vect{y}_k) + \vect{z}_k - g_k\vect{y}_k, \\
   \vect{y}_{k+1} &= (\mat{T}_k\mat{\Gamma}_k + \mat{T}_k\mat{\Gamma}_k\mat{R}_k
     + \mat{I})\,\vect{y}_k.

After :math:`d` doublings the thermal source vectors are recovered from

.. math::
   :label: yz_to_source

   s^{+}_i = y_{d,i}\,\bar{B} + z_{d,i}\,B_d, \qquad
   s^{-}_i = y_{d,i}\,\bar{B} - z_{d,i}\,B_d.

**Solar source doubling.**  With :math:`\gamma_k = \exp(-2^k\delta\tau/\mu_0)`:

.. math::

   \vect{s}^{+}_{k+1,\mathrm{sol}} &= \mat{T}_k\mat{\Gamma}_k
     (\mat{R}_k\vect{s}^{-}_{k,\mathrm{sol}} +
      \gamma_k\vect{s}^{+}_{k,\mathrm{sol}}) +
      \vect{s}^{+}_{k,\mathrm{sol}}, \\
   \vect{s}^{-}_{k+1,\mathrm{sol}} &= \mat{T}_k\mat{\Gamma}_k
     (\gamma_k\mat{R}_k\vect{s}^{+}_{k,\mathrm{sol}} +
      \vect{s}^{-}_{k,\mathrm{sol}}) +
      \gamma_k\vect{s}^{-}_{k,\mathrm{sol}}.

The attenuation factor squares at every step:
:math:`\gamma_{k+1} = \gamma_k^2`.


Pure absorption (:math:`\omega = 0`)
------------------------------------

When there is no scattering the doubling recursion is bypassed and the source
vectors are computed analytically.  With :math:`t_i = \ee^{-\tau/\mu_i}`:

.. math::

   s^{+}_i &= \bar{B}(1-t_i) + B_d\bigl[\mu_i(1-t_i) - \tfrac{\tau}{2}(1+t_i)\bigr], \\
   s^{-}_i &= \bar{B}(1-t_i) - B_d\bigl[\mu_i(1-t_i) - \tfrac{\tau}{2}(1+t_i)\bigr].

The reflection matrix is zero and the transmission is diagonal,
:math:`T_{ij} = t_i\,\delta_{ij}`, which the adding step exploits
(:doc:`adding`).
