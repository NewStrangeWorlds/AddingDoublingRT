/// @file benchmark_cuda.cu
/// @brief Performance benchmark: CPU vs CUDA adding-doubling solver.

#include "adding_doubling.hpp"
#include "cuda_solver.cuh"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
//  Build test atmosphere
// ============================================================================

struct TestAtmosphere {
  int nlay;
  int nwav;
  int nmu;
  int nmom;

  std::vector<float> delta_tau;    // [nwav * nlay]
  std::vector<float> ssa;          // [nwav * nlay]
  std::vector<float> pmom;         // [nlay * nmom] (shared)
  std::vector<float> planck;       // [nwav * nlev]
};

TestAtmosphere buildAtmosphere(int nlay, int nwav, int nmu, int nmom) {
  TestAtmosphere atm;
  atm.nlay = nlay;
  atm.nwav = nwav;
  atm.nmu = nmu;
  atm.nmom = nmom;

  int nlev = nlay + 1;

  atm.delta_tau.resize(nwav * nlay);
  atm.ssa.resize(nwav * nlay);
  atm.pmom.resize(nlay * nmom, 0.0f);
  atm.planck.resize(nwav * nlev);

  // Henyey-Greenstein g=0.7 moments: chi[l] = g^l
  double g = 0.7;
  for (int l = 0; l < nlay; ++l) {
    double gl = 1.0;
    for (int m = 0; m < nmom; ++m) {
      atm.pmom[l * nmom + m] = static_cast<float>(gl);
      gl *= g;
    }
  }

  // Per-wavenumber optical properties (vary slightly to be realistic)
  for (int w = 0; w < nwav; ++w) {
    double wfrac = static_cast<double>(w) / nwav;

    for (int l = 0; l < nlay; ++l) {
      double lfrac = static_cast<double>(l) / nlay;
      // Optical depth increases with depth, varies with wavenumber
      atm.delta_tau[w * nlay + l] = static_cast<float>(0.05 + 0.5 * lfrac + 0.1 * wfrac);
      // Single-scattering albedo
      atm.ssa[w * nlay + l] = static_cast<float>(0.7 + 0.2 * (1.0 - lfrac));
    }

    // Temperature profile: 200 K at top, 300 K at bottom
    for (int l = 0; l <= nlay; ++l) {
      double lfrac = static_cast<double>(l) / nlay;
      double T = 200.0 + 100.0 * lfrac;
      // Use Planck-like values (arbitrary units, just need nonzero)
      atm.planck[w * nlev + l] = static_cast<float>(T * T * (1.0 + 0.5 * wfrac));
    }
  }

  return atm;
}


// ============================================================================
//  CPU benchmark
// ============================================================================

double benchmarkCPU(const TestAtmosphere& atm, int nruns) {
  // Build ADConfig for a single wavenumber and run it nwav times
  // (This is how the CPU solver would be called in practice)

  adrt::SolverWorkspace ws;
  double total_flux = 0.0;

  auto t0 = Clock::now();

  for (int run = 0; run < nruns; ++run) {
    for (int w = 0; w < atm.nwav; ++w) {
      adrt::ADConfig cfg;
      cfg.num_layers = atm.nlay;
      cfg.num_quadrature = atm.nmu;
      cfg.allocate();

      for (int l = 0; l < atm.nlay; ++l) {
        cfg.delta_tau[l] = static_cast<double>(atm.delta_tau[w * atm.nlay + l]);
        cfg.single_scat_albedo[l] = static_cast<double>(atm.ssa[w * atm.nlay + l]);
      }

      // Set HG phase function
      cfg.setHenyeyGreenstein(0.7);

      // Planck levels
      int nlev = atm.nlay + 1;
      cfg.planck_levels.resize(nlev);
      for (int l = 0; l <= atm.nlay; ++l)
        cfg.planck_levels[l] = static_cast<double>(atm.planck[w * nlev + l]);

      cfg.surface_albedo = 0.1;

      auto result = adrt::solve(cfg, ws);
      total_flux += result.flux_up[0];
    }
  }

  auto t1 = Clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Prevent optimiser from eliminating the computation
  if (total_flux == -999.0) std::cout << total_flux;

  return ms / nruns;
}


// ============================================================================
//  CUDA benchmark
// ============================================================================

double benchmarkCUDA(const TestAtmosphere& atm, int nruns) {
  int nwav = atm.nwav;
  int nlay = atm.nlay;
  int nmom = atm.nmom;
  int nlev = nlay + 1;

  // Allocate device memory
  float *d_dtau, *d_ssa, *d_pmom, *d_planck;
  float *d_flux_up, *d_flux_down, *d_flux_direct;

  cudaMalloc(&d_dtau, nwav * nlay * sizeof(float));
  cudaMalloc(&d_ssa, nwav * nlay * sizeof(float));
  cudaMalloc(&d_pmom, nlay * nmom * sizeof(float));
  cudaMalloc(&d_planck, nwav * nlev * sizeof(float));
  cudaMalloc(&d_flux_up, nwav * sizeof(float));
  cudaMalloc(&d_flux_down, nwav * sizeof(float));
  cudaMalloc(&d_flux_direct, nwav * sizeof(float));

  // Upload data
  cudaMemcpy(d_dtau, atm.delta_tau.data(), nwav * nlay * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ssa, atm.ssa.data(), nwav * nlay * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_pmom, atm.pmom.data(), nlay * nmom * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_planck, atm.planck.data(), nwav * nlev * sizeof(float),
             cudaMemcpyHostToDevice);

  adrt::cuda::BatchConfig bcfg;
  bcfg.num_wavenumbers = nwav;
  bcfg.num_layers = nlay;
  bcfg.num_quadrature = atm.nmu;
  bcfg.num_moments_max = nmom;
  bcfg.surface_albedo = 0.1;

  adrt::cuda::DeviceData data;
  data.delta_tau = d_dtau;
  data.single_scat_albedo = d_ssa;
  data.phase_moments = d_pmom;
  data.phase_moments_shared = true;
  data.planck_levels = d_planck;
  data.flux_up = d_flux_up;
  data.flux_down = d_flux_down;
  data.flux_direct = d_flux_direct;

  // Warmup
  adrt::cuda::solveBatch(bcfg, data);
  cudaDeviceSynchronize();

  // Timed runs
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (int run = 0; run < nruns; ++run)
    adrt::cuda::solveBatch(bcfg, data);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms_gpu = 0.0f;
  cudaEventElapsedTime(&ms_gpu, start, stop);

  // Read back one value to verify
  float first_flux;
  cudaMemcpy(&first_flux, d_flux_up, sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_dtau);
  cudaFree(d_ssa);
  cudaFree(d_pmom);
  cudaFree(d_planck);
  cudaFree(d_flux_up);
  cudaFree(d_flux_down);
  cudaFree(d_flux_direct);

  return static_cast<double>(ms_gpu) / nruns;
}


// ============================================================================
//  Main
// ============================================================================

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::cout << "Adding-Doubling Solver Benchmark\n";
  std::cout << "================================\n";
  std::cout << "GPU: " << prop.name << "\n\n";

  std::cout << std::fixed << std::setprecision(2);

  // Benchmark configurations
  struct Config {
    int nlay, nwav, nmu, nmom;
    int cpu_runs, cuda_runs;
    const char* label;
  };

  Config configs[] = {
    {  10,   100, 8, 16,  5,  100, "Small  (10 layers, 100 wn, N=8)"},
    {  50,  1000, 8, 16,  1,  100, "Medium (50 layers, 1000 wn, N=8)"},
    { 100, 20000, 8, 16,  1,   50, "Large  (100 layers, 20000 wn, N=8)"},
    { 100, 20000, 4, 8,   1,   50, "Large  (100 layers, 20000 wn, N=4)"},
  };

  std::cout << std::left << std::setw(46) << "Configuration"
            << std::right << std::setw(12) << "CPU (ms)"
            << std::setw(12) << "CUDA (ms)"
            << std::setw(12) << "Speedup"
            << "\n";
  std::cout << std::string(82, '-') << "\n";

  for (auto& c : configs) {
    auto atm = buildAtmosphere(c.nlay, c.nwav, c.nmu, c.nmom);

    double cuda_ms = benchmarkCUDA(atm, c.cuda_runs);

    // Only run CPU benchmark for small/medium configs to avoid very long waits
    double cpu_ms = 0.0;
    bool ran_cpu = true;

    if (c.nwav <= 1000) {
      cpu_ms = benchmarkCPU(atm, c.cpu_runs);
    }
    else {
      // For large configs, benchmark CPU with fewer wavenumbers and extrapolate
      auto small_atm = buildAtmosphere(c.nlay, 100, c.nmu, c.nmom);
      double cpu_small_ms = benchmarkCPU(small_atm, 1);
      cpu_ms = cpu_small_ms * (static_cast<double>(c.nwav) / 100.0);
      ran_cpu = false;
    }

    double speedup = cpu_ms / cuda_ms;

    std::cout << std::left << std::setw(46) << c.label
              << std::right << std::setw(10) << cpu_ms
              << (ran_cpu ? "  " : " *")
              << std::setw(10) << cuda_ms << "  "
              << std::setw(10) << speedup << "x"
              << "\n";
  }

  std::cout << "\n* = CPU time extrapolated from 100-wavenumber subset\n";

  return 0;
}
