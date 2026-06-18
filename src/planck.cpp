/// @file planck.cpp
/// @brief Planck function implementation (adapted from DisORT Planck.cpp).

#include "planck.hpp"
#include "constants.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace adrt {

namespace PlanckConstants 
{
  constexpr double C2 = 1.438786;
  constexpr double SIGMA = 5.67032E-8;
  constexpr double VCUT = 1.5;
  constexpr double A1 = 1.0 / 3.0;
  constexpr double A2 = -1.0 / 8.0;
  constexpr double A3 = 1.0 / 60.0;
  constexpr double A4 = -1.0 / 5040.0;
  constexpr double A5 = 1.0 / 272160.0;
  constexpr double A6 = -1.0 / 13305600.0;
}



static double planckKernel(double v) 
{
  return v * v * v / (std::exp(v) - 1.0);
}



double planckFunction(double wnumlo, double wnumhi, double temp) 
{
  using namespace PlanckConstants;

  if (temp < 0.0 || wnumhi < wnumlo || wnumlo < 0.0)
    throw std::invalid_argument("planckFunction: invalid arguments");

  if (temp < 1.e-4)
    return 0.0;

  if (wnumhi == wnumlo) 
  {
    double arg = std::exp(-C2 * wnumhi / temp);
    return 1.1911e-8 * wnumhi * wnumhi * wnumhi * arg / (1.0 - arg);
  }

  double vmax = std::log(std::numeric_limits<double>::max());
  double sigdpi = SIGMA / PI;
  double conc = 15.0 / std::pow(PI, 4.0);

  double v[2] = { C2 * wnumlo / temp, C2 * wnumhi / temp };

  if (v[0] > std::numeric_limits<double>::epsilon() &&
      v[1] < vmax && (wnumhi - wnumlo) / wnumhi < 1.e-2) 
  {
    double hh = v[1] - v[0];
    double oldval = 0.0;
    double val0 = planckKernel(v[0]) + planckKernel(v[1]);
    
    for (int n = 1; n <= 10; ++n) 
    {
      double del = hh / (2.0 * n);
      double val = val0;
      
      for (int k = 1; k <= 2 * n - 1; ++k)
        val += 2.0 * (1 + k % 2) * planckKernel(v[0] + k * del);
      
      val *= del * A1;
      
      if (std::abs((val - oldval) / val) <= 1.e-6)
        return sigdpi * std::pow(temp, 4.0) * conc * val;
      
      oldval = val;
    }

    return sigdpi * std::pow(temp, 4.0) * conc * oldval;
  }

  double p[2] = {0.0, 0.0};
  double d[2] = {0.0, 0.0};
  int smallv = 0;
  constexpr double vcp[7] = {10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0};

  for (int i = 0; i < 2; ++i) 
  {
    if (v[i] < VCUT) 
    {
      smallv++;
      double vsq = v[i] * v[i];
      
      p[i] = conc * vsq * v[i] *
        (A1 + v[i] * (A2 + v[i] * (A3 + vsq * (A4 + vsq * (A5 + vsq * A6)))));
    }
    else 
    {
      int mmax = 1;
      
      while (v[i] < vcp[mmax - 1]) mmax++;
      
      double ex = std::exp(-v[i]);
      double exm = 1.0;
      d[i] = 0.0;
      
      for (int m = 1; m <= mmax; ++m) 
      {
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

  ans *= sigdpi * std::pow(temp, 4.0);

  return (ans == 0.0) ? 0.0 : ans;
}



double planckFunctionDeriv(double wnumlo, double wnumhi, double temp)
{
  using namespace PlanckConstants;

  if (temp < 0.0 || wnumhi < wnumlo || wnumlo < 0.0)
    throw std::invalid_argument("planckFunctionDeriv: invalid arguments");

  if (temp < 1.e-4)
    return 0.0;

  // Stable Planck shape function f(v) = v^3 / (e^v - 1), valid for all v >= 0.
  auto fkernel = [](double v) -> double
  {
    if (v < 1.e-6) return v * v;             // limit v^3/(e^v - 1) -> v^2 as v->0
    double em = std::exp(-v);
    return v * v * v * em / (1.0 - em);      // = v^3/(e^v - 1), overflow-safe
  };

  // Single-wavenumber (spectral) case: b = c1 wvn^3 / (e^x - 1), x = C2 wvn/T.
  // db/dT = b * (x/T) / (1 - e^{-x}).
  if (wnumhi == wnumlo)
  {
    double x = C2 * wnumhi / temp;
    double em = std::exp(-x);
    double b = 1.1911e-8 * wnumhi * wnumhi * wnumhi * em / (1.0 - em);
    return b * (x / temp) / (1.0 - em);
  }

  // Band-integrated case. With B = (sigma/pi)(15/pi^4) T^4 I, I = integral_{v0}^{v1} f,
  //   dB/dT = 4 B / T  -  (sigma/pi)(15/pi^4) T^3 (v1 f(v1) - v0 f(v0)).
  double sigdpi = SIGMA / PI;
  double conc = 15.0 / std::pow(PI, 4.0);
  double v0 = C2 * wnumlo / temp;
  double v1 = C2 * wnumhi / temp;

  double B = planckFunction(wnumlo, wnumhi, temp);
  double boundary = sigdpi * conc * temp * temp * temp
                    * (v1 * fkernel(v1) - v0 * fkernel(v0));

  return 4.0 * B / temp - boundary;
}

} // namespace adrt
