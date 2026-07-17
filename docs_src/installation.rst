Installation
============

AddingDoublingRT ships as three independent backends.  You only need to build
the ones you intend to use.

Requirements
------------

C++ / CUDA
~~~~~~~~~~

* A C++17 compiler (GCC, Clang, or MSVC).
* CMake ≥ 3.15.
* `Eigen 3.4 <https://eigen.tuxfamily.org/>`__ — fetched automatically by CMake
  via ``FetchContent``; no manual installation required.
* A CUDA toolkit (optional, only for the GPU backend).

JAX
~~~

* Python ≥ 3.9.
* `JAX <https://jax.readthedocs.io/>`__ (``pip install jax``).
* NumPy.
* For GPU acceleration: ``pip install "jax[cuda12]"``.


Building the C++ / CUDA backends
--------------------------------

CPU only:

.. code-block:: bash

   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build

CPU **and** CUDA:

.. code-block:: bash

   cmake -B build -DCMAKE_BUILD_TYPE=Release -DADRT_ENABLE_CUDA=ON
   cmake --build build

The build produces the example/demo program (``ad_example``), the C++ test
suite, and — when CUDA is enabled — the CPU-vs-GPU benchmark
(``ad_cuda_benchmark``).


Using the JAX backend
---------------------

The JAX backend is a plain Python package and does not need to be compiled.
Import it directly from the source tree:

.. code-block:: python

   from src_jax import ADConfig, solve
   from src_jax import BatchConfig, solve_batch

Importing the package enables 64-bit floating point in JAX
(``jax_enable_x64``), which the solver relies on for numerical stability.


Running the examples and tests
------------------------------

C++ / CUDA:

.. code-block:: bash

   # Run the example/demo program
   ./build/ad_example

   # Run the C++ test suite
   cd build && ctest --output-on-failure

   # Run the CPU-vs-CUDA benchmark
   ./build/ad_cuda_benchmark

JAX:

.. code-block:: bash

   # Run the JAX test suite (mirrors the C++ tests)
   pytest tests/test_jax_solver.py -v

   # Run the 3-way benchmark (C++ CPU vs CUDA vs JAX)
   python tests/benchmark_all.py --build-dir build


Building this documentation
---------------------------

The documentation is built with `Sphinx <https://www.sphinx-doc.org/>`__ and
the Read-the-Docs theme:

.. code-block:: bash

   pip install -r docs_src/requirements.txt
   sphinx-build -M html ./docs_src ./docs

The rendered HTML is written to ``docs/html``.  Pushing to the ``doc`` branch
triggers the GitHub Pages workflow, which runs the same command and publishes
the result.
