/// @file cuda_quadrature.cuh
/// @brief Quadrature nodes/weights and Legendre polynomials for CUDA solver.
///
/// Quadrature is precomputed on the host (in double) and uploaded to constant
/// memory as float for use by the solver kernels.

#pragma once

#include <cmath>
#include <vector>

namespace adrt {
namespace cuda {

/// Maximum quadrature order and Legendre moments supported.
constexpr int MAX_NMU = 32;
constexpr int MAX_NMOM = 128;

/// Constant memory for quadrature data (shared across all threads).
/// These are populated by the host before kernel launch.
__constant__ float d_mu[MAX_NMU];
__constant__ float d_wt[MAX_NMU];
__constant__ float d_xfac;  // 0.5 / sum(mu_i * wt_i)

/// Constant memory for precomputed Legendre polynomials P_l(mu_i).
/// Layout: d_Pl[l * MAX_NMU + i] = P_l(mu_i)
__constant__ float d_Pl[MAX_NMOM * MAX_NMU];
__constant__ int d_Pl_num_orders;  // number of precomputed orders


// ============================================================================
//  Host-side functions to compute and upload quadrature data
// ============================================================================

/// Compute Gauss-Legendre quadrature on [0, 1].
/// Identical to adrt::gaussLegendre() but standalone for CUDA host code.
inline void hostGaussLegendre(
    int n, std::vector<double>& nodes, std::vector<double>& weights)
{
  constexpr double PI = 3.14159265358979323846;
  nodes.resize(n);
  weights.resize(n);

  for (int i = 0; i < (n + 1) / 2; ++i) {
    double z = std::cos(PI * (i + 0.75) / (n + 0.5));
    double pp = 0.0;

    for (int iter = 0; iter < 100; ++iter) {
      double p0 = 1.0, p1 = z;

      for (int k = 2; k <= n; ++k) {
        double pk = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = pk;
      }

      pp = n * (z * p1 - p0) / (z * z - 1.0);
      double dz = -p1 / pp;
      z += dz;

      if (std::abs(dz) < 1e-15) break;
    }

    double w = 2.0 / ((1.0 - z * z) * pp * pp);
    int j1 = i;
    int j2 = n - 1 - i;
    nodes[j1]   = (1.0 - z) / 2.0;
    nodes[j2]   = (1.0 + z) / 2.0;
    weights[j1] = w / 2.0;
    weights[j2] = w / 2.0;
  }
}

/// Precompute Legendre polynomials P_l(x_i) for l=0..L-1.
/// Returns flat array: Pl[l * n + i] = P_l(x_i).
inline std::vector<double> hostPrecomputeLegendrePolynomials(
    int L, const std::vector<double>& x)
{
  int n = static_cast<int>(x.size());
  std::vector<double> Pl(L * n);

  for (int i = 0; i < n; ++i)
    Pl[0 * n + i] = 1.0;

  if (L > 1) {
    for (int i = 0; i < n; ++i)
      Pl[1 * n + i] = x[i];
  }

  for (int l = 2; l < L; ++l) {
    for (int i = 0; i < n; ++i)
      Pl[l * n + i] = ((2 * l - 1) * x[i] * Pl[(l - 1) * n + i]
                        - (l - 1) * Pl[(l - 2) * n + i]) / l;
  }

  return Pl;
}

/// Upload quadrature nodes, weights, and Legendre polynomials to constant memory.
/// Call this once before launching the solver kernel.
/// Host computation in double; uploaded as float.
/// @param nmu  Number of quadrature points
/// @param L    Number of Legendre orders to precompute
inline void uploadQuadratureData(int nmu, int L) {
  std::vector<double> mu, wt;
  hostGaussLegendre(nmu, mu, wt);

  double xfac_sum = 0.0;
  for (int i = 0; i < nmu; ++i)
    xfac_sum += mu[i] * wt[i];
  double xfac_d = 0.5 / xfac_sum;

  // Convert to float for upload
  std::vector<float> mu_f(nmu), wt_f(nmu);
  for (int i = 0; i < nmu; ++i) {
    mu_f[i] = static_cast<float>(mu[i]);
    wt_f[i] = static_cast<float>(wt[i]);
  }
  float xfac_f = static_cast<float>(xfac_d);

  cudaMemcpyToSymbol(d_mu, mu_f.data(), nmu * sizeof(float));
  cudaMemcpyToSymbol(d_wt, wt_f.data(), nmu * sizeof(float));
  cudaMemcpyToSymbol(d_xfac, &xfac_f, sizeof(float));

  if (L > 0) {
    auto Pl = hostPrecomputeLegendrePolynomials(L, mu);

    std::vector<float> Pl_padded(L * MAX_NMU, 0.0f);
    for (int l = 0; l < L; ++l)
      for (int i = 0; i < nmu; ++i)
        Pl_padded[l * MAX_NMU + i] = static_cast<float>(Pl[l * nmu + i]);

    cudaMemcpyToSymbol(d_Pl, Pl_padded.data(), L * MAX_NMU * sizeof(float));
    cudaMemcpyToSymbol(d_Pl_num_orders, &L, sizeof(int));
  }
}

} // namespace cuda
} // namespace adrt
