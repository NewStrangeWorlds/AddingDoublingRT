/// @file phase_matrix.cpp
/// @brief Dynamic-size phase matrix construction implementation.

#include "phase_matrix.hpp"

namespace adrt {

void computePhaseMatricesFromLegendre(
  const std::vector<double>& chi,
  const std::vector<std::vector<double>>& Pl,
  const std::vector<double>& weights,
  DynamicMatrix& Ppp,
  DynamicMatrix& Ppm)
{
  int nmu = static_cast<int>(weights.size());
  int L = static_cast<int>(chi.size());
  Ppp = DynamicMatrix(nmu);
  Ppm = DynamicMatrix(nmu);

  for (int i = 0; i < nmu; ++i) 
  {
    for (int j = 0; j < nmu; ++j) 
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

  for (int j = 0; j < nmu; ++j) 
  {
    double sum = 0.0;
    
    for (int i = 0; i < nmu; ++i)
      sum += (Ppp(i, j) + Ppm(i, j)) * weights[i];

    if (sum > 0.0) 
    {
      double correction = 1.0 / (2.0 * PI * sum);

      for (int i = 0; i < nmu; ++i) 
      {
        Ppp(i, j) *= correction;
        Ppm(i, j) *= correction;
      }
    }
  }
}



void computePhaseMatricesFromLegendre(
  const std::vector<double>& chi,
  const std::vector<double>& mu,
  const std::vector<double>& weights,
  DynamicMatrix& Ppp,
  DynamicMatrix& Ppm)
{
  int L = static_cast<int>(chi.size());
  auto Pl = precomputeLegendrePolynomials(L, mu);
  
  computePhaseMatricesFromLegendre(chi, Pl, weights, Ppp, Ppm);
}



void computeSolarPhaseVectorsDynamic(
  const std::vector<double>& chi,
  const std::vector<std::vector<double>>& Pl,
  const std::vector<double>& weights,
  double mu0,
  std::vector<double>& p_plus,
  std::vector<double>& p_minus)
{
  int nmu = static_cast<int>(weights.size());
  int L = static_cast<int>(chi.size());
  p_plus.resize(nmu);
  p_minus.resize(nmu);

  std::vector<double> Pl_mu0(L);
  Pl_mu0[0] = 1.0;

  if (L > 1) Pl_mu0[1] = mu0;
  
  for (int l = 2; l < L; ++l)
    Pl_mu0[l] = ((2 * l - 1) * mu0 * Pl_mu0[l - 1] - (l - 1) * Pl_mu0[l - 2]) / l;

  for (int i = 0; i < nmu; ++i) 
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

  for (int i = 0; i < nmu; ++i)
    sum += (p_plus[i] + p_minus[i]) * weights[i];
  
  if (sum > 0.0) 
  {
    double correction = 1.0 / (2.0 * PI * sum);
    
    for (int i = 0; i < nmu; ++i) 
    {
      p_plus[i]  *= correction;
      p_minus[i] *= correction;
    }
  }
}



void computeSolarPhaseVectorsDynamic(
  const std::vector<double>& chi,
  const std::vector<double>& mu,
  const std::vector<double>& weights,
  double mu0,
  std::vector<double>& p_plus,
  std::vector<double>& p_minus)
{
  int L = static_cast<int>(chi.size());
  auto Pl = precomputeLegendrePolynomials(L, mu);

  computeSolarPhaseVectorsDynamic(chi, Pl, weights, mu0, p_plus, p_minus);
}

} // namespace adrt
