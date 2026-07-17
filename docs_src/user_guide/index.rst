User guide
==========

This guide explains how to drive the solver: how to build a configuration,
which knobs control which physics, and how to read the results.  It is written
against the C++ and single-wavenumber JAX APIs, which share the same field
names (in ``snake_case`` for JAX and ``camelCase`` helpers in C++).  The
batched CUDA and JAX interfaces are covered in :doc:`../backends/index`.

.. toctree::
   :maxdepth: 2

   configuration
   outputs
   thermal_emission
   solar_beam
   phase_functions
   boundary_conditions
   delta_m
   jacobians
