/// @file phase_matrix.h
/// @brief Azimuthally-averaged phase matrix construction from Legendre coefficients.
///
/// Provides both fixed-size (template) and dynamic-size implementations.

#pragma once

#include "constants.hpp"
#include "matrix.hpp"
#include "quadrature.hpp"

#include <cmath>
#include <vector>

namespace adrt {

// ============================================================================
//  Dynamic-size interface (declared here, defined in phase_matrix.cpp)
// ============================================================================

/// Build azimuthally-averaged phase matrices from Legendre coefficients.
void computePhaseMatricesFromLegendre(
  const std::vector<double>& chi,
  const std::vector<double>& mu,
  const std::vector<double>& weights,
  DynamicMatrix& Ppp,
  DynamicMatrix& Ppm);

/// Build phase matrices from pre-computed Legendre polynomials (dynamic version).
void computePhaseMatricesFromLegendre(
  const std::vector<double>& chi,
  const std::vector<std::vector<double>>& Pl,
  const std::vector<double>& weights,
  DynamicMatrix& Ppp,
  DynamicMatrix& Ppm);

/// Build solar phase vectors from Legendre coefficients (dynamic version).
void computeSolarPhaseVectorsDynamic(
  const std::vector<double>& chi,
  const std::vector<double>& mu,
  const std::vector<double>& weights,
  double mu0,
  std::vector<double>& p_plus,
  std::vector<double>& p_minus);

/// Build solar phase vectors from pre-computed Legendre polynomials (dynamic version).
void computeSolarPhaseVectorsDynamic(
  const std::vector<double>& chi,
  const std::vector<std::vector<double>>& Pl,
  const std::vector<double>& weights,
  double mu0,
  std::vector<double>& p_plus,
  std::vector<double>& p_minus);


// ============================================================================
//  Fixed-size template implementations
// ============================================================================

/// Build phase matrices from pre-computed Legendre polynomials.
template<int N>
void computePhaseMatricesFromLegendreImpl(
  const std::vector<double>& chi,
  const std::vector<std::vector<double>>& Pl,
  const std::vector<double>& weights,
  Matrix<N>& Ppp,
  Matrix<N>& Ppm)
{
  int L = static_cast<int>(chi.size());
  Ppp = Matrix<N>();
  Ppm = Matrix<N>();

  for (int i = 0; i < N; ++i) 
  {
    for (int j = 0; j < N; ++j) 
    {
      double sum_pp = 0.0, sum_pm = 0.0;
      double sign = 1.0;
      
      for (int l = 0; l < L; ++l) 
      {
        double term = (2 * l + 1) * chi[l] * Pl[l][i] * Pl[l][j];
        sum_pp += term;
        sum_pm += sign * term;
        sign = -sign;
      }

      Ppp(i, j) = sum_pp / (2.0 * PI);
      Ppm(i, j) = sum_pm / (2.0 * PI);
    }
  }

  for (int j = 0; j < N; ++j) 
  {
    double sum = 0.0;
    
    for (int i = 0; i < N; ++i)
      sum += (Ppp(i, j) + Ppm(i, j)) * weights[i];

    if (sum > 0.0) 
    {
      double correction = 1.0 / (2.0 * PI * sum);
      
      for (int i = 0; i < N; ++i) 
      {
        Ppp(i, j) *= correction;
        Ppm(i, j) *= correction;
      }
    }
  }
}

/// Convenience overload: computes Legendre polynomials internally.
template<int N>
void computePhaseMatricesFromLegendreImpl(
  const std::vector<double>& chi,
  const std::vector<double>& mu,
  const std::vector<double>& weights,
  Matrix<N>& Ppp,
  Matrix<N>& Ppm)
{
  int L = static_cast<int>(chi.size());
  auto Pl = precomputeLegendrePolynomials(L, mu);

  computePhaseMatricesFromLegendreImpl<N>(chi, Pl, weights, Ppp, Ppm);
}


/// Build solar phase vectors from pre-computed Legendre polynomials.
template<int N>
void computeSolarPhaseVectorsImpl(
  const std::vector<double>& chi,
  const std::vector<std::vector<double>>& Pl,
  const std::vector<double>& weights,
  double mu0,
  typename Matrix<N>::EigenVec& p_plus,
  typename Matrix<N>::EigenVec& p_minus)
{
  using Vec = typename Matrix<N>::EigenVec;
  int L = static_cast<int>(chi.size());
  
  p_plus = Vec::Zero();
  p_minus = Vec::Zero();

  std::vector<double> Pl_mu0(L);
  Pl_mu0[0] = 1.0;
  
  if (L > 1) Pl_mu0[1] = mu0;
  
  for (int l = 2; l < L; ++l)
    Pl_mu0[l] = ((2 * l - 1) * mu0 * Pl_mu0[l - 1] - (l - 1) * Pl_mu0[l - 2]) / l;

  for (int i = 0; i < N; ++i) 
  {
    double sum_p = 0.0, sum_m = 0.0;
    double sign = 1.0;
    
    for (int l = 0; l < L; ++l) 
    {
      double term = (2 * l + 1) * chi[l] * Pl[l][i] * Pl_mu0[l];
      sum_p += term;
      sum_m += sign * term;
      sign = -sign;
    }

    p_plus[i]  = sum_p / (2.0 * PI);
    p_minus[i] = sum_m / (2.0 * PI);
  }

  double sum = 0.0;
  
  for (int i = 0; i < N; ++i)
    sum += (p_plus[i] + p_minus[i]) * weights[i];
  
  if (sum > 0.0) 
  {
    double correction = 1.0 / (2.0 * PI * sum);
    p_plus  *= correction;
    p_minus *= correction;
  }
}

/// Convenience overload: computes Legendre polynomials internally.
template<int N>
void computeSolarPhaseVectorsImpl(
  const std::vector<double>& chi,
  const std::vector<double>& mu,
  const std::vector<double>& weights,
  double mu0,
  typename Matrix<N>::EigenVec& p_plus,
  typename Matrix<N>::EigenVec& p_minus)
{
  int L = static_cast<int>(chi.size());
  auto Pl = precomputeLegendrePolynomials(L, mu);
  
  computeSolarPhaseVectorsImpl<N>(chi, Pl, weights, mu0, p_plus, p_minus);
}

} // namespace adrt
