/// @file quadrature.cpp
/// @brief Gauss-Legendre quadrature and Legendre polynomial implementation.

#include "quadrature.hpp"
#include "constants.hpp"

#include <cmath>

namespace adrt {

void gaussLegendre(
  int n,
  std::vector<double>& nodes,
  std::vector<double>& weights)
{
  nodes.resize(n);
  weights.resize(n);

  for (int i = 0; i < (n + 1) / 2; ++i) 
  {
    double z = std::cos(PI * (i + 0.75) / (n + 0.5));
    double pp = 0.0;

    for (int iter = 0; iter < 100; ++iter) 
    {
      double p0 = 1.0, p1 = z;
      
      for (int k = 2; k <= n; ++k) 
      {
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

std::vector<std::vector<double>> precomputeLegendrePolynomials(
  int L, 
  const std::vector<double>& x)
{
  int nx = static_cast<int>(x.size());
  std::vector<std::vector<double>> Pl(L, std::vector<double>(nx));

  for (int i = 0; i < nx; ++i)
    Pl[0][i] = 1.0;
  
  if (L > 1) 
  {
    for (int i = 0; i < nx; ++i)
      Pl[1][i] = x[i];
  }
  
  for (int l = 2; l < L; ++l) 
  {
    for (int i = 0; i < nx; ++i)
      Pl[l][i] = ((2 * l - 1) * x[i] * Pl[l - 1][i] - (l - 1) * Pl[l - 2][i]) / l;
  }

  return Pl;
}

} // namespace adrt
