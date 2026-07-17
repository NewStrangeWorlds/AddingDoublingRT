AddingDoublingRT
================

**AddingDoublingRT** (``adrt``) is an implementation of the adding–doubling
method for solving the radiative transfer equation in plane-parallel,
scattering, absorbing, and thermally emitting atmospheres.  It is based on the
matrix operator method of Plass, Hansen & Kattawar (1973) with the
inhomogeneous-source and delta-M extensions of Wiscombe (1975, 1976, 1977).

The solver computes hemispheric fluxes, the actinic mean intensity, the net
flux divergence (heating rate), the attenuated direct solar beam, and — for
thermal problems — the **analytic temperature Jacobians** of all of these
quantities.

Three interchangeable backends share a common configuration model and produce
identical results:

* **C++ (CPU)** — template-optimised kernels for :math:`N = 2, 4, 8, 16, 32`
  quadrature streams with a dynamic fallback for arbitrary :math:`N`.
* **CUDA (GPU)** — a batched solver that processes every wavenumber of a
  spectrum in parallel, one thread per wavenumber.
* **JAX (CPU/GPU)** — a pure-Python implementation that is fully JIT-compiled,
  vectorised across wavenumbers, and differentiable through
  :func:`jax.grad`.

.. note::

   This documentation covers the physics and mathematics of the method
   (:doc:`theory/index`), how to drive the solver (:doc:`user_guide/index`),
   the three backends and their APIs (:doc:`backends/index`), and the source
   references (:doc:`references`).  If you are new to the code, start with
   :doc:`quickstart`.


Feature overview
----------------

* Thermal emission with a linear-in-optical-depth Planck source (Wiscombe 1976).
* A collimated solar/stellar beam tracked separately through the doubling and
  adding steps, with direct-beam attenuation and diffuse scattering.
* Combined thermal + solar sources in a single solve.
* Delta-M scaling (Wiscombe 1977) for strongly forward-peaked phase functions.
* A Lambertian surface with configurable albedo and thermal emission, or a
  diffusion-approximation lower boundary for optically thick stellar atmospheres.
* Built-in phase functions — isotropic, Rayleigh, Henyey–Greenstein, double
  Henyey–Greenstein — or arbitrary Legendre moments.
* Analytic temperature Jacobians of the fluxes, mean intensity, and heating
  rate (C++ backend), computed by reusing the forward operators.


.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Backends & API

   backends/index

.. toctree::
   :maxdepth: 1
   :caption: Appendix

   references


Indices
-------

* :ref:`genindex`
* :ref:`search`
