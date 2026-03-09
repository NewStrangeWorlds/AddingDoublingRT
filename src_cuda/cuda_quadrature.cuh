/// @file cuda_quadrature.cuh
/// @brief Quadrature nodes/weights and Legendre polynomials for CUDA solver.
///
/// Quadrature is precomputed on the host and uploaded to constant memory.
/// Legendre polynomials at the quadrature nodes are also precomputed.

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
__constant__ double d_mu[MAX_NMU];
__constant__ double d_wt[MAX_NMU];
__constant__ double d_xfac;  // 0.5 / sum(mu_i * wt_i)

/// Constant memory for precomputed Legendre polynomials P_l(mu_i).
/// Layout: d_Pl[l * MAX_NMU + i] = P_l(mu_i)
__constant__ double d_Pl[MAX_NMOM * MAX_NMU];
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
/// @param nmu  Number of quadrature points
/// @param L    Number of Legendre orders to precompute
inline void uploadQuadratureData(int nmu, int L) {
  std::vector<double> mu, wt;
  hostGaussLegendre(nmu, mu, wt);

  double xfac_sum = 0.0;
  for (int i = 0; i < nmu; ++i)
    xfac_sum += mu[i] * wt[i];
  double xfac = 0.5 / xfac_sum;

  cudaMemcpyToSymbol(d_mu, mu.data(), nmu * sizeof(double));
  cudaMemcpyToSymbol(d_wt, wt.data(), nmu * sizeof(double));
  cudaMemcpyToSymbol(d_xfac, &xfac, sizeof(double));

  if (L > 0) {
    auto Pl = hostPrecomputeLegendrePolynomials(L, mu);
    // Pl is [L][nmu] contiguous — but constant memory expects [MAX_NMU] stride.
    // Copy row by row if nmu < MAX_NMU.
    std::vector<double> Pl_padded(L * MAX_NMU, 0.0);
    for (int l = 0; l < L; ++l)
      for (int i = 0; i < nmu; ++i)
        Pl_padded[l * MAX_NMU + i] = Pl[l * nmu + i];

    cudaMemcpyToSymbol(d_Pl, Pl_padded.data(), L * MAX_NMU * sizeof(double));
    cudaMemcpyToSymbol(d_Pl_num_orders, &L, sizeof(int));
  }
}

} // namespace cuda
} // namespace adrt
