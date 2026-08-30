/// @file doubling.hpp
/// @brief Doubling algorithm: compute R, T, source vectors for a single
///        homogeneous layer via iterative doubling.

#pragma once

#include "constants.hpp"
#include "layer.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace adrt {

/// Adaptive number of initial doublings based on single-scattering albedo.
/// The thin-layer start is exact through first order in the scattering and
/// second order in the multiple-scattering expansion, so the count needed for
/// a given accuracy grows with omega; weakly scattering layers need fewer.
inline int computeIpow0(double omega) 
{
  if (omega < 0.01) return 4;
  if (omega < 0.1)  return 10;
  
  return 16;
}


/// Number of doublings for a layer of optical depth tau.
///
/// Two criteria are combined:
///   1. the omega-adaptive multiple-scattering criterion, ipow0(omega) + log2(tau);
///   2. an extinction floor, tau0 = tau / 2^nn <= mu_min / 2, which keeps the
///      Taylor-expanded double-scattering terms of the start (polynomial in
///      omega*tau0/mu) inside their convergence regime. Without it, thick and
///      weakly scattering layers (small ipow0) would start from tau0 >> mu_min.
///
/// The three ipow0 values can be overridden (the CUDA backends use smaller ones).
inline int computeDoublingCount(
  double tau, double omega, double mu_min,
  int ipow0_weak = 4, int ipow0_mid = 10, int ipow0_strong = 16)
{
  int ipow0 = (omega < 0.01) ? ipow0_weak : (omega < 0.1) ? ipow0_mid : ipow0_strong;
  int nn = static_cast<int>(std::log(tau) / std::log(2.0)) + ipow0;

  int n_ext = static_cast<int>(std::ceil(std::log2(tau / mu_min))) + 1;

  return std::max({1, nn, n_ext});
}


/// phi(u) = expm1(u) / u, with phi(0) = 1.
inline double doublingPhi(double u)
{
  return (std::fabs(u) < 1e-9) ? 1.0 + 0.5 * u : std::expm1(u) / u;
}


/// int_0^tau0 exp(-(tau0-t)/mu_i) exp(-t/mu_j) dt = (e_j - e_i) / (1/mu_i - 1/mu_j)
/// with e_k = exp(-tau0/mu_k) and d = 1/mu_i - 1/mu_j. Evaluated as
/// tau0 e_i phi(tau0 d) for small |tau0 d| (no cancellation) and as the
/// difference form otherwise (no overflow of expm1 for large tau0/mu).
inline double doublingPathIntegral(double tau0, double d, double e_i, double e_j)
{
  const double u = tau0 * d;
  return (std::fabs(u) < 1.0) ? tau0 * e_i * doublingPhi(u) : (e_j - e_i) / d;
}


/// Thin-layer initialisation of the doubling recursion.
///
/// For a homogeneous sub-layer of optical depth tau0 with scattering operators
///   Spp(i,j) = 2*pi*omega * Ppp(i,j) * w_j / mu_i   (forward,  same hemisphere)
///   Spm(i,j) = 2*pi*omega * Ppm(i,j) * w_j / mu_i   (backward, opposite hemisphere)
/// the start is
///   T(i,j) = delta_ij e_i + Spp(i,j) * int_0^tau0 exp(-(tau0-t)/mu_i) exp(-t/mu_j) dt
///          + tau0^2/2 * (Spp^2 + Spm^2)(i,j)
///   R(i,j) = Spm(i,j) * int_0^tau0 exp(-t/mu_i) exp(-t/mu_j) dt
///          + tau0^2/2 * (Spp*Spm + Spm*Spp)(i,j)
/// with e_i = exp(-tau0/mu_i): exact extinction, exact single scattering and
/// Taylor-expanded double scattering, i.e. an O(tau0^3) start. The extinction
/// and single-scattering terms are exact for any tau0/mu, so the start never
/// leaves the physical range (the former first-order start, T = I - tau0*Gpp,
/// has a negative diagonal for tau0 > mu_min and blows up under doubling).
///
/// The thermal source is the absorbed fraction of the same operators
/// (Kirchhoff), consistent through O(tau0^2):
///   y_i = (1-omega) * [ (1 - e_i) + tau0^2/2 * sum_k (Spp+Spm)(i,k) / mu_k ]
/// so that sum_j (T+R)(i,j) + y_i = 1 + O(tau0^3) and y == 0 for omega = 1.
/// z (the linear-in-tau source gradient term) is the pure-absorption value,
/// O(tau0^3). The solar source uses the same exact-single + Taylor-double
/// treatment with the direct beam attenuated along its own path.
///
/// Templated on the matrix type so the fixed-size (Matrix<N>) and dynamic
/// (DynamicMatrix) code paths share one implementation. Vectors are accessed
/// via operator[] only.
template<class Mat, class Vec>
void initDoublingStart(
  Mat& R, Mat& T, Vec& y, Vec& z,
  Vec& s_up_sol, Vec& s_down_sol,
  const int n,
  const double tau0, const double omega,
  const Mat& Spp, const Mat& Spm,
  const std::vector<double>& mu,
  const bool has_solar, const double F_top, const double solar_mu,
  const double* p_plus_solar, const double* p_minus_solar)
{
  const double h = 0.5 * tau0 * tau0;

  std::vector<double> inv_mu(n), e(n), a(n);
  
  for (int i = 0; i < n; ++i) 
  {
    inv_mu[i] = 1.0 / mu[i];
    const double x = tau0 * inv_mu[i];
    e[i] = (x < 700.0) ? std::exp(-x) : 0.0;
    a[i] = -std::expm1(-x);       // 1 - e_i, accurate for small x
  }

  // --- exact extinction + exact single scattering ---
  for (int i = 0; i < n; ++i) 
  {
    for (int j = 0; j < n; ++j) 
    {
      const double d = inv_mu[i] - inv_mu[j];     // T: exp(-(tau0-t)/mu_i) exp(-t/mu_j)
      const double sdiff = inv_mu[i] + inv_mu[j]; // R: exp(-t/mu_i) exp(-t/mu_j)

      const double int_T = doublingPathIntegral(tau0, d, e[i], e[j]);
      const double int_R = tau0 * doublingPhi(-tau0 * sdiff);

      T(i, j) = ((i == j) ? e[i] : 0.0) + Spp(i, j) * int_T;
      R(i, j) = Spm(i, j) * int_R;
    }
  }

  // --- Taylor double scattering ---
  const Mat SppSpp = Spp.multiply(Spp);
  const Mat SpmSpm = Spm.multiply(Spm);
  const Mat SppSpm = Spp.multiply(Spm);
  const Mat SpmSpp = Spm.multiply(Spp);

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) 
    {
      T(i, j) += h * (SppSpp(i, j) + SpmSpm(i, j));
      R(i, j) += h * (SppSpm(i, j) + SpmSpp(i, j));
    }

  // --- thermal source ---
  for (int i = 0; i < n; ++i) 
  {
    double d2 = 0.0;
    
    for (int k = 0; k < n; ++k)
      d2 += (Spp(i, k) + Spm(i, k)) * inv_mu[k];

    y[i] = (1.0 - omega) * (a[i] + h * d2);

    // z_i = (1-omega) * [ mu_i (1 - e_i) - tau0/2 (1 + e_i) ]  (= -(1-omega) mu_i x^3/12 + ...)
    const double x = tau0 * inv_mu[i];
    double slope;
    
    if (x < 1e-3)
      slope = mu[i] * x * x * x * (-1.0 / 12.0 + x * (1.0 / 24.0 - x / 80.0));
    else
      slope = mu[i] * a[i] - 0.5 * tau0 * (1.0 + e[i]);

    z[i] = (1.0 - omega) * slope;
  }

  // --- solar source ---
  if (has_solar) 
  {
    const double inv_mu0 = 1.0 / solar_mu;
    const double e0 = (tau0 * inv_mu0 < 700.0) ? std::exp(-tau0 * inv_mu0) : 0.0;
    std::vector<double> src_up(n), src_dn(n);   // single-scatter source per unit tau
    
    for (int i = 0; i < n; ++i) 
    {
      src_up[i] = omega * F_top * p_minus_solar[i] * inv_mu[i];
      src_dn[i] = omega * F_top * p_plus_solar[i]  * inv_mu[i];
    }

    for (int i = 0; i < n; ++i) 
    {
      // up: int_0^tau0 exp(-t/mu0) exp(-t/mu_i) dt ; down: int_0^tau0 exp(-t/mu0) exp(-(tau0-t)/mu_i) dt
      const double int_up = tau0 * doublingPhi(-tau0 * (inv_mu0 + inv_mu[i]));
      const double int_dn = doublingPathIntegral(tau0, inv_mu[i] - inv_mu0, e[i], e0);

      double d_up = 0.0, d_dn = 0.0;
      
      for (int k = 0; k < n; ++k) 
      {
        d_up += Spp(i, k) * src_up[k] + Spm(i, k) * src_dn[k];
        d_dn += Spp(i, k) * src_dn[k] + Spm(i, k) * src_up[k];
      }

      s_up_sol[i]   = src_up[i] * int_up + h * d_up;
      s_down_sol[i] = src_dn[i] * int_dn + h * d_dn;
    }
  } 
  else 
  {
    for (int i = 0; i < n; ++i) 
    {
      s_up_sol[i] = 0.0;
      s_down_sol[i] = 0.0;
    }
  }
}


template<int N>
LayerMatrices<N> doubling(
  double tau,
  double omega,
  double B_top,
  double B_bottom,
  const Matrix<N>& Ppp,
  const Matrix<N>& Ppm,
  const std::vector<double>& mu,
  const std::vector<double>& weights,
  double solar_flux = 0.0,
  double solar_mu = 0.0,
  double tau_cumulative = 0.0,
  const typename Matrix<N>::EigenVec* p_plus_solar = nullptr,
  const typename Matrix<N>::EigenVec* p_minus_solar = nullptr,
  int nn_override = -1)
{
  using Vec = typename Matrix<N>::EigenVec;

  LayerMatrices<N> layer;

  double B_bar = (B_bottom + B_top) / 2.0;
  double B_d = (tau > 0.0) ? (B_bottom - B_top) / tau : 0.0;

  if (tau <= 0.0)
    return layer;

  // Pure absorption
  if (omega <= 0.0) 
  {
    layer.T_ab = Matrix<N>();
    layer.T_ba = Matrix<N>();

    for (int i = 0; i < N; ++i) 
    {
      double tex = -tau / mu[i];
      double trans = (tex > -200.0) ? std::exp(tex) : 0.0;
      layer.T_ab(i, i) = trans;
      layer.T_ba(i, i) = trans;
      double one_minus_t = 1.0 - trans;
      double slope_term = mu[i] * one_minus_t - 0.5 * tau * (1.0 + trans);
      layer.s_up[i]   = B_bar * one_minus_t + B_d * slope_term;
      layer.s_down[i] = B_bar * one_minus_t - B_d * slope_term;
      // Source-derivative basis (tau > 0 here): a_i = one_minus_t, b_i = slope_term.
      layer.j_p[i] = 0.5 * one_minus_t - slope_term / tau;
      layer.j_q[i] = 0.5 * one_minus_t + slope_term / tau;
    }

    return layer;
  }

  // General case: scattering layer
  layer.is_scattering = true;
  omega = std::clamp(omega, 0.0, 1.0);

  double con = 2.0 * omega * PI;

  Vec wt_vec;
  for (int i = 0; i < N; ++i)
    wt_vec[i] = weights[i];

  auto I = Matrix<N>::identity();
  Matrix<N> C = Matrix<N>::diagonal(wt_vec);

  // Scattering operators Spp = 2 pi omega Ppp C / mu,  Spm = 2 pi omega Ppm C / mu
  Matrix<N> PppC = Ppp.multiply(C);
  Matrix<N> PpmC = Ppm.multiply(C);
  Matrix<N> Spp, Spm;

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) 
    {
      Spp(i, j) = con * PppC(i, j) / mu[i];
      Spm(i, j) = con * PpmC(i, j) / mu[i];
    }

  const double mu_min = *std::min_element(mu.begin(), mu.begin() + N);
  int nn = (nn_override >= 0) ? nn_override : computeDoublingCount(tau, omega, mu_min);
  
  if (nn < 1) nn = 1;
  
  double xfac = 1.0 / std::pow(2.0, nn);
  double tau0 = tau * xfac;

  bool has_solar = (solar_flux > 0.0 && solar_mu > 0.0
                    && p_plus_solar != nullptr && p_minus_solar != nullptr);
  double F_top = has_solar ? solar_flux * std::exp(-tau_cumulative / solar_mu) : 0.0;

  Matrix<N> R_k, T_k;
  Vec y_k, z_k;
  Vec s_up_sol_k, s_down_sol_k;

  initDoublingStart(R_k, T_k, y_k, z_k, s_up_sol_k, s_down_sol_k,
                    N, tau0, omega, Spp, Spm, mu,
                    has_solar, F_top, solar_mu,
                    has_solar ? p_plus_solar->data() : nullptr,
                    has_solar ? p_minus_solar->data() : nullptr);

  double g_k = 0.5 * tau0;
  double gamma_sol = has_solar ? std::exp(-tau0 / solar_mu) : 0.0;

  for (int k = 0; k < nn; ++k) 
  {
    Matrix<N> R_sq = R_k.multiply(R_k);
    Matrix<N> I_minus_R2 = I.add(R_sq, -1.0);

    Matrix<N> TG = I_minus_R2.rightSolveMatrix(T_k);
    Matrix<N> TGR = TG.multiply(R_k);

    Matrix<N> R_new = R_k.add(TGR.multiply(T_k));
    Matrix<N> T_new = TG.multiply(T_k);

    Vec zpgy = z_k + g_k * y_k;

    Vec TG_zpgy  = TG.multiply(zpgy);
    Vec TGR_zpgy = TGR.multiply(zpgy);
    Vec TG_y  = TG.multiply(y_k);
    Vec TGR_y = TGR.multiply(y_k);

    Vec z_new, y_new;

    for (int i = 0; i < N; ++i) 
    {
      z_new[i] = (TG_zpgy[i] - TGR_zpgy[i]) + z_k[i] - g_k * y_k[i];
      y_new[i] = TG_y[i] + TGR_y[i] + y_k[i];
    }

    Vec s_up_sol_new = Vec::Zero(), s_down_sol_new = Vec::Zero();
    
    if (has_solar) 
    {
      Vec R_sdown = R_k.multiply(s_down_sol_k);
      Vec R_sup   = R_k.multiply(s_up_sol_k);

      Vec rhs_up, rhs_down;
      
      for (int i = 0; i < N; ++i) 
      {
        rhs_up[i]   = R_sdown[i] + gamma_sol * s_up_sol_k[i];
        rhs_down[i] = gamma_sol * R_sup[i] + s_down_sol_k[i];
      }

      Vec TG_rhs_up   = TG.multiply(rhs_up);
      Vec TG_rhs_down = TG.multiply(rhs_down);

      for (int i = 0; i < N; ++i) 
      {
        s_up_sol_new[i]   = TG_rhs_up[i] + s_up_sol_k[i];
        s_down_sol_new[i] = TG_rhs_down[i] + gamma_sol * s_down_sol_k[i];
      }

      gamma_sol = gamma_sol * gamma_sol;
    }

    R_k = std::move(R_new);
    T_k = std::move(T_new);
    y_k = std::move(y_new);
    z_k = std::move(z_new);
    s_up_sol_k   = std::move(s_up_sol_new);
    s_down_sol_k = std::move(s_down_sol_new);
    g_k = 2.0 * g_k;
  }

  LayerMatrices<N> result;
  result.is_scattering = true;
  result.R_ab = R_k;
  result.R_ba = R_k;
  result.T_ab = T_k;
  result.T_ba = T_k;

  for (int i = 0; i < N; ++i) 
  {
    result.s_up[i]   = y_k[i] * B_bar + z_k[i] * B_d;
    result.s_down[i] = y_k[i] * B_bar - z_k[i] * B_d;
    result.s_up_solar[i]   = s_up_sol_k[i];
    result.s_down_solar[i] = s_down_sol_k[i];
    // Source-derivative basis (tau > 0 here): a_i = y_k, b_i = z_k.
    result.j_p[i] = 0.5 * y_k[i] - z_k[i] / tau;
    result.j_q[i] = 0.5 * y_k[i] + z_k[i] / tau;
  }

  return result;
}

} // namespace adrt
