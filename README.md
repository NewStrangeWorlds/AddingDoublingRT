# AddingDoublingRT

An implementation of the adding-doubling method for solving the radiative transfer equation in plane-parallel atmospheres. Based on the matrix operator method of Plass, Hansen & Kattawar (1973, *Applied Optics* 12, 314). Available in three backends: **C++ CPU**, **CUDA GPU**, and **JAX** (CPU/GPU).

## Features

- **Thermal emission** with linear-in-optical-depth Planck source interpolation
- **Solar/stellar beam** with direct beam attenuation and diffuse scattering
- **Combined sources** (thermal + solar) for realistic atmosphere modelling
- **Delta-M scaling** (Wiscombe 1977) for accelerated convergence with forward-peaked phase functions
- **Lambertian surface** with configurable albedo and emission
- **Diffusion approximation** lower boundary condition for stellar atmosphere models
- Built-in phase functions: isotropic, Rayleigh, Henyey-Greenstein, double Henyey-Greenstein, or arbitrary Legendre moments

### C++ CPU solver

- Template-optimised kernels for N = 2, 4, 8, 16, 32 quadrature points (Eigen fixed-size matrices, stack allocation, SIMD) with automatic fallback to dynamic-size matrices for arbitrary N
- `SolverWorkspace` for caching Legendre polynomials across repeated calls

### CUDA GPU solver

- Batched solver processing all wavenumbers in parallel (one CUDA thread per wavenumber)
- Template-specialised kernels (`solveKernel<N>`) for N = 2, 4, 8, 16 with register-resident matrix operations
- Double-precision LU factorisation for linear systems
- Shared or per-wavenumber phase function moments

### JAX solver

- Pure-Python implementation using JAX for automatic differentiation and GPU acceleration
- **Single-wavenumber solver** (`solve`) matching the C++ API
- **Batched solver** (`solve_batch`) vectorised across wavenumbers via `jnp.einsum` and `jax.lax.scan`
- Float64 precision by default
- Full JIT compilation into a single XLA program

## Output

The solver returns upward and downward diffuse fluxes, the attenuated direct beam flux, and the mean intensity at each layer interface (from the top of the atmosphere to the surface).

## Requirements

### C++ / CUDA

- C++17 compiler (GCC, Clang, or MSVC)
- CMake >= 3.15
- [Eigen 3.4](https://eigen.tuxfamily.org/) (fetched automatically via CMake FetchContent)
- CUDA toolkit (optional, for GPU solver)

### JAX

- Python >= 3.9
- JAX (`pip install jax`)
- NumPy
- For GPU acceleration: `pip install jax[cuda12]`

## Building

### C++ CPU only

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### C++ CPU + CUDA

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DADRT_ENABLE_CUDA=ON
cmake --build build
```

## Running the examples and tests

### C++ / CUDA

```bash
# Run the example/demo program
./build/ad_example

# Run the C++ test suite
cd build && ctest --output-on-failure

# Run the performance benchmark (CPU vs CUDA)
./build/ad_cuda_benchmark
```

### JAX

```bash
# Run the JAX test suite (63 tests)
pytest tests/test_jax_solver.py -v

# Run the 3-way performance benchmark (C++ CPU vs CUDA vs JAX)
python tests/benchmark_all.py --build-dir build
```

## Quick start

### C++ API

```cpp
#include "adding_doubling.hpp"

// Set up a 5-layer atmosphere with 8-stream quadrature
adrt::ADConfig cfg(5, 8);
cfg.solar_flux = 1.0;
cfg.solar_mu   = 0.5;        // cos(solar zenith angle)
cfg.surface_albedo = 0.3;
cfg.allocate();

for (int l = 0; l < 5; ++l) {
    cfg.delta_tau[l] = 0.2;
    cfg.single_scat_albedo[l] = 0.9;
}
cfg.setHenyeyGreenstein(0.7);

adrt::RTOutput result = adrt::solve(cfg);
// result.flux_up[0]  -> upward flux at TOA
// result.flux_down[5] -> downward flux at BOA
```

For thermal emission, set `use_thermal_emission = true`, provide level temperatures and wavenumber bounds, then call `allocate()`:

```cpp
adrt::ADConfig cfg(10, 8);
cfg.use_thermal_emission = true;
cfg.wavenumber_low  = 500.0;   // cm^-1
cfg.wavenumber_high = 1500.0;
cfg.allocate();

// Fill cfg.temperature[0..10], cfg.delta_tau, cfg.single_scat_albedo, ...
adrt::RTOutput result = adrt::solve(cfg);
```

### JAX API (single wavenumber)

```python
from src_jax import ADConfig, solve

cfg = ADConfig()
cfg.num_layers = 5
cfg.num_quadrature = 8
cfg.solar_flux = 1.0
cfg.solar_mu = 0.5
cfg.surface_albedo = 0.3
cfg.allocate()

for l in range(5):
    cfg.delta_tau[l] = 0.2
    cfg.single_scat_albedo[l] = 0.9
cfg.set_henyey_greenstein(0.7)

result = solve(cfg)
# result.flux_up[0]  -> upward flux at TOA
# result.flux_down[5] -> downward flux at BOA
```

### JAX API (batched across wavenumbers)

```python
import numpy as np
from src_jax import BatchConfig, solve_batch

bcfg = BatchConfig()
bcfg.num_wavenumbers = 1000
bcfg.num_layers = 50
bcfg.num_quadrature = 8
bcfg.num_moments_max = 16
bcfg.surface_albedo = 0.1
bcfg.solar_flux = 1.0    # optional: solar beam
bcfg.solar_mu = 0.5

delta_tau = np.random.uniform(0.01, 0.5, (1000, 50))
ssa = np.full((1000, 50), 0.9)
pmom = np.zeros((50, 16))  # shared across wavenumbers
for l in range(50):
    for m in range(16):
        pmom[l, m] = 0.7 ** m  # Henyey-Greenstein g=0.7
planck = np.zeros((1000, 51))  # zero = no thermal emission

flux_up, flux_down = solve_batch(bcfg, delta_tau, ssa, pmom, planck)
# flux_up.shape  -> (1000,)  TOA upward flux per wavenumber
# flux_down.shape -> (1000,) TOA downward flux per wavenumber
```

## Project structure

```
src/
  adding_doubling.hpp  -- Public API: ADConfig, RTOutput, solve()
  solver.cpp           -- Solver dispatch and templated implementation
  adding.hpp           -- Adding step: combine two layers
  doubling.hpp         -- Doubling step: build single-layer R/T matrices
  phase_matrix.hpp/cpp -- Phase matrix construction from Legendre moments
  quadrature.hpp/cpp   -- Gauss-Legendre quadrature and Legendre polynomials
  planck.hpp/cpp       -- Planck function integration
  layer.hpp            -- Layer data structures (fixed-size and dynamic)
  matrix.hpp           -- Dense NxN matrix wrappers around Eigen
  workspace.hpp        -- Reusable workspace for Legendre caching
  constants.hpp        -- Shared constants
  example.cpp          -- Example/demo program
src_cuda/
  cuda_solver.cuh      -- Public CUDA API: BatchConfig, DeviceData, solveBatch()
  cuda_solver.cu       -- Kernel dispatch and template instantiation
  cuda_matrix.cuh      -- GpuMatrix<N>/GpuVec<N>, LU solve (double precision)
  cuda_doubling.cuh    -- Doubling iteration (register-resident)
  cuda_adding.cuh      -- Layer combination
  cuda_layer.cuh       -- Per-thread layer data structures
  cuda_phase_matrix.cuh -- Phase matrix construction on device
  cuda_quadrature.cuh  -- Gauss-Legendre quadrature (device constant memory)
  cuda_planck.cuh      -- Planck function on device
src_jax/
  __init__.py          -- Package init (enables float64)
  config.py            -- ADConfig and RTOutput dataclasses
  solver.py            -- Single-wavenumber solver
  batch_solver.py      -- Batched solver (vectorised across wavenumbers)
  doubling.py          -- Doubling step
  adding.py            -- Adding step
  phase_matrix.py      -- Phase matrix construction
  quadrature.py        -- Gauss-Legendre quadrature and Legendre polynomials
  planck.py            -- Planck function
  example.py           -- Example/demo program
tests/
  testing.hpp          -- Lightweight C++ test framework
  test_ad_solver.cpp   -- C++ solver test suite (63 tests)
  test_jax_solver.py   -- JAX solver test suite (63 tests, mirrors C++ tests)
  benchmark_cuda.cu    -- C++ CPU vs CUDA benchmark
  benchmark_all.py     -- 3-way benchmark (C++ CPU vs CUDA vs JAX)
```

## References

- Plass, G. N., Kattawar, G. W., & Catchings, F. E. (1973). Matrix operator theory of radiative transfer. 1: Rayleigh scattering. *Applied Optics*, 12(2), 314-329.
- Hansen, J. E. (1971). Multiple scattering of polarized light in planetary atmospheres. Part II. Sunlight reflected by terrestrial water clouds. *Journal of the Atmospheric Sciences*, 28(8), 1400-1426.
- Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate radiative flux calculations for strongly asymmetric phase functions. *Journal of the Atmospheric Sciences*, 34(9), 1408-1422.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
