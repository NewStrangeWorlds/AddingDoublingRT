Theory
======

This section describes the mathematics of the adding–doubling solver.  It
follows Plass, Hansen & Kattawar (1973) and Wiscombe (1975, 1976, 1977), with
the extensions used in ``adrt``:

#. a linear-in-:math:`\tau` thermal source (Wiscombe 1976);
#. separate tracking of the direct solar beam through the doubling and adding
   steps;
#. delta-M scaling for forward-peaked phase functions (Wiscombe 1977);
#. a diffusion-approximation lower boundary condition for stellar atmospheres;
#. analytic temperature Jacobians obtained by re-running the forward operators.

Throughout, the solver uses Gauss–Legendre quadrature on :math:`[0, 1]` with
:math:`M` streams per hemisphere.  All matrices are :math:`M \times M` and all
source vectors have length :math:`M`.  The azimuthally averaged
(:math:`m = 0`) problem is solved, which is sufficient for hemispheric fluxes
and mean intensities.

.. toctree::
   :maxdepth: 2

   overview
   doubling
   adding
   boundary_intensity
   phase_function
   delta_m
   jacobian
   implementation
