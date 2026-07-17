Phase functions
===============

Phase functions are specified through their **reduced** Legendre moments
:math:`\chi_\ell` (:doc:`../theory/phase_function`).  The zeroth moment is the
normalisation and must be :math:`1`; ``allocate()`` sets it for you.  The
per-layer array ``phase_function_moments[l]`` holds :math:`2M` moments (or
:math:`2M+1` when delta-M is enabled).


Built-in helpers
----------------

Each helper takes an optional layer index; ``-1`` (the default) applies to all
layers.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - C++ / JAX
     - Effect
   * - ``setIsotropic()`` / ``set_isotropic()``
     - :math:`\chi_0 = 1`, all higher moments zero.
   * - ``setRayleigh()`` / ``set_rayleigh()``
     - :math:`\chi_2 = 0.1`, others zero.
   * - ``setHenyeyGreenstein(g)`` / ``set_henyey_greenstein(g)``
     - :math:`\chi_\ell = g^\ell`.
   * - ``setDoubleHenyeyGreenstein(f, g1, g2)`` /
       ``set_double_henyey_greenstein(f, g1, g2)``
     - :math:`\chi_\ell = f\,g_1^\ell + (1-f)\,g_2^\ell`.

.. code-block:: cpp

   cfg.setHenyeyGreenstein(0.7);       // all layers, g = 0.7
   cfg.setRayleigh(0);                 // layer 0 only
   cfg.setDoubleHenyeyGreenstein(0.9, 0.8, -0.5, 3);  // layer 3


Arbitrary phase functions
-------------------------

Fill the moment array directly.  Only supply as many moments as you have; the
rest stay zero:

.. code-block:: cpp

   // Custom expansion for layer l
   for (int k = 0; k < n_moments; ++k)
     cfg.phase_function_moments[l][k] = chi[k];

.. code-block:: python

   import numpy as np
   cfg.phase_function_moments[l] = np.array(chi)   # length up to 2*M (+1)

.. warning::

   Keep :math:`\chi_0 = 1`.  If you plan to use delta-M scaling, provide the
   :math:`2M`-th moment as well (index ``2*num_quadrature``), since it defines
   the truncation fraction :math:`f` (:doc:`delta_m`).
