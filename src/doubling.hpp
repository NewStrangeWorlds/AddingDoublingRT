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
/// Single-scattering error is O(tau0^2 * omega^2), so weaker scattering
/// needs fewer doublings to reach the same accuracy.
inline int computeIpow0(double omega) 
{
  if (omega < 0.01) return 4;
  if (omega < 0.1)  return 10;
  
  return 16;
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
  const typename Matrix<N>::EigenVec* p_minus_solar = nullptr)
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

  Matrix<N> PppC = Ppp.multiply(C);
  Matrix<N> temp = I.add(PppC, -con);
  Matrix<N> Gpp;

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      Gpp(i, j) = temp(i, j) / mu[i];

  Matrix<N> PpmC = Ppm.multiply(C);
  Matrix<N> Gpm;
  
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      Gpm(i, j) = con * PpmC(i, j) / mu[i];

  int nn = static_cast<int>(std::log(tau) / std::log(2.0)) + computeIpow0(omega);
  
  if (nn < 1) nn = 1;
  
  double xfac = 1.0 / std::pow(2.0, nn);
  double tau0 = tau * xfac;

  bool has_solar = (solar_flux > 0.0 && solar_mu > 0.0
                    && p_plus_solar != nullptr && p_minus_solar != nullptr);
  double F_top = has_solar ? solar_flux * std::exp(-tau_cumulative / solar_mu) : 0.0;

  Matrix<N> R_k, T_k;
  
  for (int i = 0; i < N; ++i) 
  {
    for (int j = 0; j < N; ++j) 
    {
      T_k(i, j) = ((i == j) ? 1.0 : 0.0) - tau0 * Gpp(i, j);
      R_k(i, j) = tau0 * Gpm(i, j);
    }
  }

  Vec y_k, z_k = Vec::Zero();

  for (int i = 0; i < N; ++i)
    y_k[i] = (1.0 - omega) * tau0 / mu[i];

  Vec s_up_sol_k = Vec::Zero(), s_down_sol_k = Vec::Zero();
  
  if (has_solar) 
  {
    for (int i = 0; i < N; ++i) 
    {
      double base = omega * tau0 / mu[i] * F_top;
      s_up_sol_k[i]   = base * (*p_minus_solar)[i];
      s_down_sol_k[i] = base * (*p_plus_solar)[i];
    }
  }

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
  }

  return result;
}

} // namespace adrt
