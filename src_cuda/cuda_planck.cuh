/// @file cuda_planck.cuh
/// @brief Device-side Planck function for the CUDA RT solver.
///
/// Port of planck.cpp adapted from DisORT Planck.cpp.

#pragma once

#include <cmath>

namespace adrt {
namespace cuda {

namespace PlanckConstants {
  __device__ constexpr double C2 = 1.438786;
  __device__ constexpr double SIGMA = 5.67032e-8;
  __device__ constexpr double VCUT = 1.5;
  __device__ constexpr double D_PI = 3.14159265358979323846;
  __device__ constexpr double A1 = 1.0 / 3.0;
  __device__ constexpr double A2 = -1.0 / 8.0;
  __device__ constexpr double A3 = 1.0 / 60.0;
  __device__ constexpr double A4 = -1.0 / 5040.0;
  __device__ constexpr double A5 = 1.0 / 272160.0;
  __device__ constexpr double A6 = -1.0 / 13305600.0;
}

__device__ __forceinline__ double planck_kernel(double v) {
  return v * v * v / (exp(v) - 1.0);
}

/// Compute Planck function integrated between two wavenumbers.
/// Device-side equivalent of adrt::planckFunction().
__device__ __forceinline__ double planck_function(
    double wnumlo, double wnumhi, double temp)
{
  using namespace PlanckConstants;

  if (temp < 1.0e-4)
    return 0.0;

  if (wnumhi == wnumlo) {
    double arg = exp(-C2 * wnumhi / temp);
    return 1.1911e-8 * wnumhi * wnumhi * wnumhi * arg / (1.0 - arg);
  }

  constexpr double VMAX = 709.0;  // ~log(DBL_MAX)
  double sigdpi = SIGMA / D_PI;
  double pi4 = D_PI * D_PI * D_PI * D_PI;
  double conc = 15.0 / pi4;

  double v0 = C2 * wnumlo / temp;
  double v1 = C2 * wnumhi / temp;

  // Simpson quadrature for narrow bands
  if (v0 > 1.0e-16 && v1 < VMAX && (wnumhi - wnumlo) / wnumhi < 1.0e-2) {
    double hh = v1 - v0;
    double oldval = 0.0;
    double val0 = planck_kernel(v0) + planck_kernel(v1);

    for (int n = 1; n <= 10; ++n) {
      double del = hh / (2.0 * n);
      double val = val0;

      for (int k = 1; k <= 2 * n - 1; ++k)
        val += 2.0 * (1 + k % 2) * planck_kernel(v0 + k * del);

      val *= del * A1;

      if (fabs((val - oldval) / val) <= 1.0e-6)
        return sigdpi * pow(temp, 4.0) * conc * val;

      oldval = val;
    }

    return sigdpi * pow(temp, 4.0) * conc * oldval;
  }

  // General case: series expansion
  double v[2] = {v0, v1};
  double p[2] = {0.0, 0.0};
  double d[2] = {0.0, 0.0};
  int smallv = 0;
  constexpr double vcp[7] = {10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0};

  for (int i = 0; i < 2; ++i) {
    if (v[i] < VCUT) {
      smallv++;
      double vsq = v[i] * v[i];
      p[i] = conc * vsq * v[i] *
        (A1 + v[i] * (A2 + v[i] * (A3 + vsq * (A4 + vsq * (A5 + vsq * A6)))));
    }
    else {
      int mmax = 1;
      while (v[i] < vcp[mmax - 1]) mmax++;

      double ex = exp(-v[i]);
      double exm = 1.0;
      d[i] = 0.0;

      for (int m = 1; m <= mmax; ++m) {
        double mv = m * v[i];
        exm *= ex;
        d[i] += exm * (6.0 + mv * (6.0 + mv * (3.0 + mv))) / (m * m * m * m);
      }

      d[i] *= conc;
    }
  }

  double ans;

  if (smallv == 2)
    ans = p[1] - p[0];
  else if (smallv == 1)
    ans = 1.0 - p[0] - d[1];
  else
    ans = d[0] - d[1];

  ans *= sigdpi * pow(temp, 4.0);

  return (ans == 0.0) ? 0.0 : ans;
}

} // namespace cuda
} // namespace adrt
