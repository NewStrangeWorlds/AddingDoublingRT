# AddingDoublingRT

A C++17 implementation of the adding-doubling method for solving the radiative transfer equation in plane-parallel atmospheres. Based on the matrix operator method of Plass, Hansen & Kattawar (1973, *Applied Optics* 12, 314).

## Features

- **Thermal emission** with linear-in-optical-depth Planck source interpolation
- **Solar/stellar beam** with direct beam attenuation and diffuse scattering
- **Delta-M scaling** (Wiscombe 1977) for accelerated convergence with forward-peaked phase functions
- **Lambertian surface** with configurable albedo and emission
- **Diffusion approximation** lower boundary condition for stellar atmosphere models
- Built-in phase functions: isotropic, Rayleigh, Henyey-Greenstein, double Henyey-Greenstein, or arbitrary Legendre moments
- **Template-optimised kernels** for N = 2, 4, 8, 16, 32 quadrature points (Eigen fixed-size matrices, stack allocation, SIMD) with automatic fallback to dynamic-size matrices for arbitrary N
- **SolverWorkspace** for caching Legendre polynomials across repeated calls

## Output

The solver returns upward and downward diffuse fluxes, the attenuated direct beam flux, and the mean intensity at each layer interface (from the top of the atmosphere to the surface).

## Requirements

- C++17 compiler (GCC, Clang, or MSVC)
- CMake >= 3.15
- [Eigen 3.4](https://eigen.tuxfamily.org/) (fetched automatically via CMake FetchContent)

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Running the examples and tests

```bash
# Run the example/demo program
./ad_example

# Run the test suite
ctest --output-on-failure
```

## Quick start

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
tests/
  testing.hpp          -- Lightweight test framework
  test_ad_solver.cpp   -- Solver test suite
```

## References

- Plass, G. N., Kattawar, G. W., & Catchings, F. E. (1973). Matrix operator theory of radiative transfer. 1: Rayleigh scattering. *Applied Optics*, 12(2), 314-329.
- Hansen, J. E. (1971). Multiple scattering of polarized light in planetary atmospheres. Part II. Sunlight reflected by terrestrial water clouds. *Journal of the Atmospheric Sciences*, 28(8), 1400-1426.
- Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate radiative flux calculations for strongly asymmetric phase functions. *Journal of the Atmospheric Sciences*, 34(9), 1408-1422.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
