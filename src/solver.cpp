/// @file solver.cpp
/// @brief Main solver: templated solveImpl, dynamic fallback, and public dispatch.

#include "adding_doubling.hpp"
#include "adding.hpp"
#include "constants.hpp"
#include "doubling.hpp"
#include "layer.hpp"
#include "phase_matrix.hpp"
#include "planck.hpp"
#include "quadrature.hpp"
#include "workspace.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace adrt {

// ============================================================================
//  Internal intensity computation (fixed-size)
// ============================================================================

template<int N>
static typename Matrix<N>::EigenVec computeIup(
  const LayerMatrices<N>& top,
  const LayerMatrices<N>& base,
  const typename Matrix<N>::EigenVec& I_top_down,
  const typename Matrix<N>::EigenVec& I_bot_up)
{
  using Vec = typename Matrix<N>::EigenVec;

  auto I = Matrix<N>::identity();
  Matrix<N> to_invert = I.add(base.R_ab.multiply(top.R_ba), -1.0);

  Vec term1 = base.T_ba.multiply(I_bot_up);
  Vec TtopI = top.T_ab.multiply(I_top_down);
  Vec term2 = base.R_ab.multiply(TtopI);
  Vec s_down_total = top.s_down + top.s_down_solar;
  Vec term3 = base.R_ab.multiply(s_down_total);

  Vec rhs = term1 + term2 + term3 + base.s_up + base.s_up_solar;

  return to_invert.solve(rhs);
}


template<int N>
static typename Matrix<N>::EigenVec computeIdown(
  const LayerMatrices<N>& top,
  const LayerMatrices<N>& base,
  const typename Matrix<N>::EigenVec& I_top_down,
  const typename Matrix<N>::EigenVec& I_bot_up)
{
  using Vec = typename Matrix<N>::EigenVec;

  auto I = Matrix<N>::identity();
  Matrix<N> to_invert = I.add(top.R_ba.multiply(base.R_ab), -1.0);

  Vec term1 = top.T_ab.multiply(I_top_down);
  Vec TbaseI = base.T_ba.multiply(I_bot_up);
  Vec term2 = top.R_ba.multiply(TbaseI);
  Vec s_up_total = base.s_up + base.s_up_solar;
  Vec term3 = top.R_ba.multiply(s_up_total);

  Vec rhs = term1 + term2 + term3 + top.s_down + top.s_down_solar;

  return to_invert.solve(rhs);
}


// ============================================================================
//  Templated solver implementation
// ============================================================================

template<int N>
static RTOutput solveImpl(const ADConfig& config, SolverWorkspace* ws) 
{
  using Vec = typename Matrix<N>::EigenVec;

  SolverWorkspace local_ws;
  if (!ws) ws = &local_ws;

  ADConfig cfg = config;
  cfg.validate();

  int nlay = cfg.num_layers;

  // Reverse arrays if indexed from bottom
  if (cfg.index_from_bottom) 
  {
    std::reverse(cfg.delta_tau.begin(), cfg.delta_tau.end());
    std::reverse(cfg.single_scat_albedo.begin(), cfg.single_scat_albedo.end());
    std::reverse(cfg.phase_function_moments.begin(), cfg.phase_function_moments.end());

    if (cfg.use_thermal_emission)
      std::reverse(cfg.temperature.begin(), cfg.temperature.end());
    if (!cfg.planck_levels.empty())
      std::reverse(cfg.planck_levels.begin(), cfg.planck_levels.end());
  }

  // --- 1. Gauss-Legendre quadrature on [0, 1] ---
  std::vector<double> mu, wt;
  gaussLegendre(N, mu, wt);

  double xfac_sum = 0.0;

  for (int i = 0; i < N; ++i)
    xfac_sum += mu[i] * wt[i];
  
  double xfac = 0.5 / xfac_sum;

  // --- 2. Compute Planck values at each level ---
  std::vector<double> B(nlay + 1, 0.0);
  double B_surface = 0.0;
  double B_top_emission = 0.0;

  if (cfg.use_thermal_emission) 
  {
    for (int l = 0; l <= nlay; ++l)
      B[l] = planckFunction(cfg.wavenumber_low, cfg.wavenumber_high, cfg.temperature[l]);
    
    B_surface = B[nlay];
    B_top_emission = B[0];
  }
  else if (static_cast<int>(cfg.planck_levels.size()) == nlay + 1) 
  {
    B = cfg.planck_levels;
    B_surface = cfg.surface_emission;
    B_top_emission = cfg.top_emission;
  }
  else 
  {
    B_surface = cfg.surface_emission;
    B_top_emission = cfg.top_emission;
  }

  // --- 3. Compute per-layer R, T, s ---
  bool has_solar = (cfg.solar_flux > 0.0 && cfg.solar_mu > 0.0);
  int two_M = 2 * N;

  std::vector<LayerMatrices<N>> layer_rtj;
  layer_rtj.reserve(nlay);
  std::vector<double> tau_used(nlay);

  double tau_cumulative = 0.0;
  for (int l = 0; l < nlay; ++l) {
    double tau_layer = cfg.delta_tau[l];
    double omega_layer = cfg.single_scat_albedo[l];

    double B_layer_top = B[l];
    double B_layer_bot = B[l + 1];

    Matrix<N> Ppp, Ppm;
    Vec p_plus_solar, p_minus_solar;
    bool has_solar_phase = false;

    if (omega_layer > 0.0 && tau_layer > 0.0) 
    {
      const auto& chi_full = cfg.phase_function_moments[l];

      if (cfg.use_delta_m) {
        double f_trunc = (static_cast<int>(chi_full.size()) > two_M) ? chi_full[two_M] : 0.0;

        if (f_trunc > 1e-12 && f_trunc < 1.0 - 1e-12) 
        {
          double omega_f = omega_layer * f_trunc;
          tau_layer   = (1.0 - omega_f) * cfg.delta_tau[l];
          omega_layer = omega_layer * (1.0 - f_trunc) / (1.0 - omega_f);

          std::vector<double> chi_star(two_M);
          double inv_1mf = 1.0 / (1.0 - f_trunc);

          for (int ll = 0; ll < two_M; ++ll)
            chi_star[ll] = (chi_full[ll] - f_trunc) * inv_1mf;

          const auto& Pl = ws->getLegendrePolynomials(two_M, mu);
          computePhaseMatricesFromLegendreImpl<N>(chi_star, Pl, wt, Ppp, Ppm);
          
          if (has_solar) {
            computeSolarPhaseVectorsImpl<N>(
                chi_star, Pl, wt, cfg.solar_mu, p_plus_solar, p_minus_solar);
            has_solar_phase = true;
          }
        }
        else 
        {
          std::vector<double> chi(chi_full.begin(),
              chi_full.begin() + std::min(static_cast<int>(chi_full.size()), two_M));
          const auto& Pl = ws->getLegendrePolynomials(
              static_cast<int>(chi.size()), mu);
          
          computePhaseMatricesFromLegendreImpl<N>(chi, Pl, wt, Ppp, Ppm);
          
          if (has_solar) 
          {
            computeSolarPhaseVectorsImpl<N>(
                chi, Pl, wt, cfg.solar_mu, p_plus_solar, p_minus_solar);
            
            has_solar_phase = true;
          }
        }
      }
      else 
      {
        const auto& Pl = ws->getLegendrePolynomials(
          static_cast<int>(chi_full.size()), mu);
        
        computePhaseMatricesFromLegendreImpl<N>(chi_full, Pl, wt, Ppp, Ppm);
        
        if (has_solar) 
        {
          computeSolarPhaseVectorsImpl<N>(
              chi_full, Pl, wt, cfg.solar_mu, p_plus_solar, p_minus_solar);
          
          has_solar_phase = true;
        }
      }
    }

    tau_used[l] = tau_layer;

    layer_rtj.push_back(
        doubling<N>(tau_layer, omega_layer,
                    B_layer_top, B_layer_bot, Ppp, Ppm, mu, wt,
                    cfg.solar_flux, cfg.solar_mu, tau_cumulative,
                    has_solar_phase ? &p_plus_solar : nullptr,
                    has_solar_phase ? &p_minus_solar : nullptr));

    tau_cumulative += tau_layer;
  }

  // --- 4. Lambertian surface layer ---
  int ltot = nlay;
  bool has_surface = !cfg.use_diffusion_lower_bc && (cfg.surface_albedo > 0.0 || B_surface > 0.0);

  if (has_surface) 
  {
    LayerMatrices<N> surf;
    surf.T_ab = Matrix<N>();
    surf.T_ba = Matrix<N>();

    double A = cfg.surface_albedo;
    
    for (int i = 0; i < N; ++i) 
    {
      for (int j = 0; j < N; ++j) 
      {
        double r = 2.0 * A * mu[j] * wt[j] * xfac;
        surf.R_ab(i, j) = r;
        surf.R_ba(i, j) = r;
      }
      surf.s_up[i]   = (1.0 - A) * B_surface;
      surf.s_down[i] = 0.0;

      if (has_solar && A > 0.0) 
      {
        surf.s_up_solar[i] = (A / PI) * cfg.solar_flux * cfg.solar_mu
                             * std::exp(-tau_cumulative / cfg.solar_mu);
      }
    }

    if (A > 0.0)
      surf.is_scattering = true;

    layer_rtj.push_back(std::move(surf));
    ltot++;
  }

  // --- 5. Build composites from bottom (RBASE) ---
  std::vector<LayerMatrices<N>> rbase;
  rbase.reserve(ltot + 1);
  rbase.emplace_back();

  rbase.push_back(layer_rtj[ltot - 1]);

  for (int l = 1; l < ltot; ++l) 
  {
    int k = ltot - 1 - l;
    rbase.push_back(addLayers<N>(layer_rtj[k], rbase[l]));
  }

  // --- 6. Build composites from top (RTOP) ---
  std::vector<LayerMatrices<N>> rtop;
  rtop.reserve(ltot + 1);
  rtop.emplace_back();

  rtop.push_back(layer_rtj[0]);

  for (int l = 1; l < ltot; ++l)
    rtop.push_back(addLayers<N>(rtop[l], layer_rtj[l]));

  // --- 7. Boundary intensities ---
  Vec I_top_down;
  for (int i = 0; i < N; ++i)
    I_top_down[i] = B_top_emission;

  Vec I_bot_up = Vec::Zero();
  if (cfg.use_diffusion_lower_bc) 
  {
    double B_bottom = B[nlay];
    double dtau_last = tau_used[nlay - 1];
    double dB_dtau = (dtau_last > 0.0) ? (B_bottom - B[nlay - 1]) / dtau_last : 0.0;
    for (int i = 0; i < N; ++i)
      I_bot_up[i] = B_bottom + mu[i] * dB_dtau;
  }
  else if (!has_surface) 
  {
    for (int i = 0; i < N; ++i)
      I_bot_up[i] = B_surface;
  }

  // --- 8. Compute intensities at each interface ---
  int n_interfaces = nlay + 1;

  RTOutput result;
  result.flux_up.resize(n_interfaces, 0.0);
  result.flux_down.resize(n_interfaces, 0.0);
  result.mean_intensity.resize(n_interfaces, 0.0);
  result.flux_direct.resize(n_interfaces, 0.0);

  if (cfg.solar_flux > 0.0 && cfg.solar_mu > 0.0) 
  {
    double tau_cum = 0.0;
    result.flux_direct[0] = cfg.solar_flux * cfg.solar_mu;
    
    for (int l = 0; l < nlay; ++l) 
    {
      tau_cum += tau_used[l];
      result.flux_direct[l + 1] = cfg.solar_flux * cfg.solar_mu
                                  * std::exp(-tau_cum / cfg.solar_mu);
    }
  }

  // Top of atmosphere
  {
    auto& full = rbase[ltot];
    Vec Iup = full.R_ab.multiply(I_top_down);
    Vec Tbot = full.T_ba.multiply(I_bot_up);
    Iup += Tbot + full.s_up + full.s_up_solar;

    for (int i = 0; i < N; ++i) 
    {
      result.flux_up[0]        += 2.0 * PI * wt[i] * mu[i] * Iup[i];
      result.flux_down[0]      += 2.0 * PI * wt[i] * mu[i] * I_top_down[i];
      result.mean_intensity[0] += 0.5 * wt[i] * (Iup[i] + I_top_down[i]);
    }
  }

  // Internal interfaces
  for (int l = 1; l <= nlay; ++l) 
  {
    int n_top = l;
    int n_base = ltot - l;

    Vec Iup, Idown;

    if (n_base > 0 && n_top > 0) 
    {
      Iup   = computeIup<N>(rtop[n_top], rbase[n_base], I_top_down, I_bot_up);
      Idown = computeIdown<N>(rtop[n_top], rbase[n_base], I_top_down, I_bot_up);
    }
    else if (n_base == 0) 
    {
      Idown = rtop[n_top].T_ab.multiply(I_top_down);
      Vec RtopIbot = rtop[n_top].R_ba.multiply(I_bot_up);
      Idown += RtopIbot + rtop[n_top].s_down + rtop[n_top].s_down_solar;
      Iup = I_bot_up;
    }
    else 
    {
      Iup = Vec::Zero();
      Idown = I_top_down;
    }

    for (int i = 0; i < N; ++i) 
    {
      result.flux_up[l]        += 2.0 * PI * wt[i] * mu[i] * Iup[i];
      result.flux_down[l]      += 2.0 * PI * wt[i] * mu[i] * Idown[i];
      result.mean_intensity[l] += 0.5 * wt[i] * (Iup[i] + Idown[i]);
    }
  }

  // Reverse output if indexed from bottom
  if (cfg.index_from_bottom) 
  {
    std::reverse(result.flux_up.begin(), result.flux_up.end());
    std::reverse(result.flux_down.begin(), result.flux_down.end());
    std::reverse(result.mean_intensity.begin(), result.mean_intensity.end());
    std::reverse(result.flux_direct.begin(), result.flux_direct.end());
  }

  return result;
}


// ============================================================================
//  Dynamic fallback helpers
// ============================================================================

static void dynAddSources(
    DynLayerMatrices& ans,
    int n,
    const DynLayerMatrices& top,
    const DynLayerMatrices& bot,
    const DynamicMatrix& T_ba_D1,
    const DynamicMatrix& T_bc_D2)
{
  auto do_sources = [&](
      const std::vector<double>& s1_up,
      const std::vector<double>& s1_down,
      const std::vector<double>& s2_up,
      const std::vector<double>& s2_down,
      std::vector<double>& ans_up,
      std::vector<double>& ans_down)
  {
    auto Rbc_s1m = bot.R_ab.multiply(s1_down);
    std::vector<double> rhs(n);
    for (int i = 0; i < n; ++i) rhs[i] = s2_up[i] + Rbc_s1m[i];
    auto contrib = T_ba_D1.multiply(rhs);
    for (int i = 0; i < n; ++i) ans_up[i] = s1_up[i] + contrib[i];

    auto Rba_s2p = top.R_ba.multiply(s2_up);
    for (int i = 0; i < n; ++i) rhs[i] = s1_down[i] + Rba_s2p[i];
    contrib = T_bc_D2.multiply(rhs);
    for (int i = 0; i < n; ++i) ans_down[i] = s2_down[i] + contrib[i];
  };

  do_sources(top.s_up, top.s_down, bot.s_up, bot.s_down,
             ans.s_up, ans.s_down);
  do_sources(top.s_up_solar, top.s_down_solar, bot.s_up_solar, bot.s_down_solar,
             ans.s_up_solar, ans.s_down_solar);
}



static DynLayerMatrices dynAddLayersGeneral(
  const DynLayerMatrices& top, 
  const DynLayerMatrices& bot)
{
  int n = top.R_ab.size();
  DynLayerMatrices ans(n);
  ans.is_scattering = true;

  auto I = DynamicMatrix::identity(n);
  DynamicMatrix A1 = I.add(bot.R_ab.multiply(top.R_ba), -1.0);
  DynamicMatrix A2 = I.add(top.R_ba.multiply(bot.R_ab), -1.0);
  DynamicMatrix T_ba_D1 = A1.rightSolveMatrix(top.T_ba);
  DynamicMatrix T_bc_D2 = A2.rightSolveMatrix(bot.T_ab);

  DynamicMatrix tmp1 = T_ba_D1.multiply(bot.R_ab);
  ans.R_ab = top.R_ab.add(tmp1.multiply(top.T_ab));
  DynamicMatrix tmp2 = T_bc_D2.multiply(top.R_ba);
  ans.R_ba = bot.R_ba.add(tmp2.multiply(bot.T_ba));
  ans.T_ab = T_bc_D2.multiply(top.T_ab);
  ans.T_ba = T_ba_D1.multiply(bot.T_ba);

  dynAddSources(ans, n, top, bot, T_ba_D1, T_bc_D2);
  return ans;
}



static DynLayerMatrices dynAddLayersNonscatTop(
  const DynLayerMatrices& top, 
  const DynLayerMatrices& bot)
{
  int n = top.R_ab.size();
  DynLayerMatrices ans(n);
  ans.is_scattering = bot.is_scattering;

  std::vector<double> t(n);
  for (int i = 0; i < n; ++i) t[i] = top.T_ab(i, i);

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      ans.R_ab(i, j) = t[i] * bot.R_ab(i, j) * t[j];
  
  ans.R_ba = bot.R_ba;
  ans.T_ab = DynamicMatrix(n);
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      ans.T_ab(i, j) = bot.T_ab(i, j) * t[j];
  
  ans.T_ba = DynamicMatrix(n);
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      ans.T_ba(i, j) = t[i] * bot.T_ba(i, j);

  auto do_sources = [&](
      const std::vector<double>& s1_up,
      const std::vector<double>& s1_down,
      const std::vector<double>& s2_up,
      const std::vector<double>& s2_down,
      std::vector<double>& ans_up,
      std::vector<double>& ans_down)
  {
    auto Rbc_s1m = bot.R_ab.multiply(s1_down);
    
    for (int i = 0; i < n; ++i)
      ans_up[i] = s1_up[i] + t[i] * (s2_up[i] + Rbc_s1m[i]);
    
      auto Tbc_s1m = bot.T_ab.multiply(s1_down);
    
    for (int i = 0; i < n; ++i)
      ans_down[i] = s2_down[i] + Tbc_s1m[i];
  };

  do_sources(top.s_up, top.s_down, bot.s_up, bot.s_down,
             ans.s_up, ans.s_down);
  do_sources(top.s_up_solar, top.s_down_solar, bot.s_up_solar, bot.s_down_solar,
             ans.s_up_solar, ans.s_down_solar);
  return ans;
}



static DynLayerMatrices dynAddLayersNonscatBot(
  const DynLayerMatrices& top, 
  const DynLayerMatrices& bot)
{
  int n = top.R_ab.size();
  DynLayerMatrices ans(n);
  ans.is_scattering = top.is_scattering;

  std::vector<double> t(n);
  for (int i = 0; i < n; ++i) t[i] = bot.T_ab(i, i);

  ans.R_ab = top.R_ab;
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      ans.R_ba(i, j) = t[i] * top.R_ba(i, j) * t[j];
  
  ans.T_ab = DynamicMatrix(n);
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      ans.T_ab(i, j) = t[i] * top.T_ab(i, j);
  
  ans.T_ba = DynamicMatrix(n);
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      ans.T_ba(i, j) = top.T_ba(i, j) * t[j];

  auto do_sources = [&](
    const std::vector<double>& s1_up,
    const std::vector<double>& s1_down,
    const std::vector<double>& s2_up,
    const std::vector<double>& s2_down,
    std::vector<double>& ans_up,
    std::vector<double>& ans_down)
  {
    auto Tba_s2p = top.T_ba.multiply(s2_up);
    
    for (int i = 0; i < n; ++i)
      ans_up[i] = s1_up[i] + Tba_s2p[i];
    
    auto Rba_s2p = top.R_ba.multiply(s2_up);
    
    for (int i = 0; i < n; ++i)
      ans_down[i] = s2_down[i] + t[i] * (s1_down[i] + Rba_s2p[i]);
  };

  do_sources(top.s_up, top.s_down, bot.s_up, bot.s_down,
             ans.s_up, ans.s_down);
  do_sources(top.s_up_solar, top.s_down_solar, bot.s_up_solar, bot.s_down_solar,
             ans.s_up_solar, ans.s_down_solar);
  return ans;
}



static DynLayerMatrices dynAddLayers(
  const DynLayerMatrices& top, 
  const DynLayerMatrices& bot)
{
  if (!top.is_scattering) return dynAddLayersNonscatTop(top, bot);
  if (!bot.is_scattering) return dynAddLayersNonscatBot(top, bot);
  
  return dynAddLayersGeneral(top, bot);
}



static DynLayerMatrices dynDoubling(
  double tau, double omega, double B_top, double B_bottom,
  const DynamicMatrix& Ppp, const DynamicMatrix& Ppm,
  const std::vector<double>& mu,
  const std::vector<double>& weights,
  double solar_flux, double solar_mu, double tau_cumulative,
  const std::vector<double>& p_plus_solar,
  const std::vector<double>& p_minus_solar)
{
  int n = static_cast<int>(mu.size());
  DynLayerMatrices layer(n);

  double B_bar = (B_bottom + B_top) / 2.0;
  double B_d = (tau > 0.0) ? (B_bottom - B_top) / tau : 0.0;

  if (tau <= 0.0) return layer;

  if (omega <= 0.0) 
  {
    layer.T_ab = DynamicMatrix(n);
    layer.T_ba = DynamicMatrix(n);
    
    for (int i = 0; i < n; ++i) 
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

  layer.is_scattering = true;
  omega = std::clamp(omega, 0.0, 1.0);
  double con = 2.0 * omega * PI;

  auto I = DynamicMatrix::identity(n);
  DynamicMatrix C = DynamicMatrix::diagonal(
    std::vector<double>(weights.begin(), 
    weights.end()));

  DynamicMatrix PppC = Ppp.multiply(C);
  DynamicMatrix temp = I.add(PppC, -con);
  DynamicMatrix Gpp(n);
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      Gpp(i, j) = temp(i, j) / mu[i];

  DynamicMatrix PpmC = Ppm.multiply(C);
  DynamicMatrix Gpm(n);
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      Gpm(i, j) = con * PpmC(i, j) / mu[i];

  int nn = static_cast<int>(std::log(tau) / std::log(2.0)) + computeIpow0(omega);
  
  if (nn < 1) nn = 1;
  
  double xfac_d = 1.0 / std::pow(2.0, nn);
  double tau0 = tau * xfac_d;

  bool has_solar = (solar_flux > 0.0 && solar_mu > 0.0
                    && !p_plus_solar.empty() && !p_minus_solar.empty());
  double F_top = has_solar ? solar_flux * std::exp(-tau_cumulative / solar_mu) : 0.0;

  DynamicMatrix R_k(n), T_k(n);
  
  for (int i = 0; i < n; ++i) 
  {
    for (int j = 0; j < n; ++j) {
      T_k(i, j) = ((i == j) ? 1.0 : 0.0) - tau0 * Gpp(i, j);
      R_k(i, j) = tau0 * Gpm(i, j);
    }
  }

  std::vector<double> y_k(n), z_k(n, 0.0);
  
  for (int i = 0; i < n; ++i)
    y_k[i] = (1.0 - omega) * tau0 / mu[i];

  std::vector<double> s_up_sol_k(n, 0.0), s_down_sol_k(n, 0.0);
  
  if (has_solar) 
  {
    for (int i = 0; i < n; ++i) {
      double base = omega * tau0 / mu[i] * F_top;
      s_up_sol_k[i]   = base * p_minus_solar[i];
      s_down_sol_k[i] = base * p_plus_solar[i];
    }
  }

  double g_k = 0.5 * tau0;
  double gamma_sol = has_solar ? std::exp(-tau0 / solar_mu) : 0.0;

  for (int k = 0; k < nn; ++k) 
  {
    DynamicMatrix R_sq = R_k.multiply(R_k);
    DynamicMatrix I_minus_R2 = I.add(R_sq, -1.0);
    DynamicMatrix TG = I_minus_R2.rightSolveMatrix(T_k);
    DynamicMatrix TGR = TG.multiply(R_k);
    DynamicMatrix R_new = R_k.add(TGR.multiply(T_k));
    DynamicMatrix T_new = TG.multiply(T_k);

    std::vector<double> z_new(n), y_new(n);
    std::vector<double> zpgy(n);

    for (int i = 0; i < n; ++i) zpgy[i] = z_k[i] + g_k * y_k[i];

    auto TG_zpgy  = TG.multiply(zpgy);
    auto TGR_zpgy = TGR.multiply(zpgy);
    auto TG_y  = TG.multiply(y_k);
    auto TGR_y = TGR.multiply(y_k);

    for (int i = 0; i < n; ++i) 
    {
      z_new[i] = (TG_zpgy[i] - TGR_zpgy[i]) + z_k[i] - g_k * y_k[i];
      y_new[i] = TG_y[i] + TGR_y[i] + y_k[i];
    }

    std::vector<double> s_up_sol_new(n, 0.0), s_down_sol_new(n, 0.0);
    
    if (has_solar) 
    {
      auto R_sdown = R_k.multiply(s_down_sol_k);
      auto R_sup   = R_k.multiply(s_up_sol_k);
      std::vector<double> rhs_up(n), rhs_down(n);
      
      for (int i = 0; i < n; ++i) 
      {
        rhs_up[i]   = R_sdown[i] + gamma_sol * s_up_sol_k[i];
        rhs_down[i] = gamma_sol * R_sup[i] + s_down_sol_k[i];
      }
      
      auto TG_rhs_up   = TG.multiply(rhs_up);
      auto TG_rhs_down = TG.multiply(rhs_down);
      
      for (int i = 0; i < n; ++i) 
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

  DynLayerMatrices result(n);
  result.is_scattering = true;
  result.R_ab = R_k;
  result.R_ba = R_k;
  result.T_ab = T_k;
  result.T_ba = T_k;

  for (int i = 0; i < n; ++i) 
  {
    result.s_up[i]   = y_k[i] * B_bar + z_k[i] * B_d;
    result.s_down[i] = y_k[i] * B_bar - z_k[i] * B_d;
    result.s_up_solar[i]   = s_up_sol_k[i];
    result.s_down_solar[i] = s_down_sol_k[i];
  }

  return result;
}


// ============================================================================
//  Dynamic fallback solver
// ============================================================================

static RTOutput solveDynamic(
  const ADConfig& config, 
  SolverWorkspace* ws) 
{
  SolverWorkspace local_ws;
  if (!ws) ws = &local_ws;

  ADConfig cfg = config;
  cfg.validate();

  int nlay = cfg.num_layers;
  int nmu  = cfg.num_quadrature;

  if (cfg.index_from_bottom) 
  {
    std::reverse(cfg.delta_tau.begin(), cfg.delta_tau.end());
    std::reverse(cfg.single_scat_albedo.begin(), cfg.single_scat_albedo.end());
    std::reverse(cfg.phase_function_moments.begin(), cfg.phase_function_moments.end());
    
    if (cfg.use_thermal_emission)
      std::reverse(cfg.temperature.begin(), cfg.temperature.end());
    
    if (!cfg.planck_levels.empty())
      std::reverse(cfg.planck_levels.begin(), cfg.planck_levels.end());
  }

  std::vector<double> mu, wt;
  gaussLegendre(nmu, mu, wt);

  double xfac_sum = 0.0;
  
  for (int i = 0; i < nmu; ++i)
    xfac_sum += mu[i] * wt[i];
  
    double xfac = 0.5 / xfac_sum;

  std::vector<double> B(nlay + 1, 0.0);
  double B_surface = 0.0;
  double B_top_emission = 0.0;

  if (cfg.use_thermal_emission) 
  {
    for (int l = 0; l <= nlay; ++l)
      B[l] = planckFunction(cfg.wavenumber_low, cfg.wavenumber_high, cfg.temperature[l]);
    
    B_surface = B[nlay];
    B_top_emission = B[0];
  }
  else if (static_cast<int>(cfg.planck_levels.size()) == nlay + 1) 
  {
    B = cfg.planck_levels;
    B_surface = cfg.surface_emission;
    B_top_emission = cfg.top_emission;
  }
  else 
  {
    B_surface = cfg.surface_emission;
    B_top_emission = cfg.top_emission;
  }

  bool has_solar = (cfg.solar_flux > 0.0 && cfg.solar_mu > 0.0);
  int two_M = 2 * nmu;

  std::vector<DynLayerMatrices> layer_rtj;
  layer_rtj.reserve(nlay);
  std::vector<double> tau_used(nlay);

  double tau_cumulative = 0.0;

  for (int l = 0; l < nlay; ++l) 
  {
    double tau_layer = cfg.delta_tau[l];
    double omega_layer = cfg.single_scat_albedo[l];
    double B_layer_top = B[l];
    double B_layer_bot = B[l + 1];

    DynamicMatrix Ppp(nmu), Ppm(nmu);
    std::vector<double> p_plus_solar, p_minus_solar;

    if (omega_layer > 0.0 && tau_layer > 0.0) 
    {
      const auto& chi_full = cfg.phase_function_moments[l];

      if (cfg.use_delta_m) 
      {
        double f_trunc = (static_cast<int>(chi_full.size()) > two_M) ? chi_full[two_M] : 0.0;
        
        if (f_trunc > 1e-12 && f_trunc < 1.0 - 1e-12) 
        {
          double omega_f = omega_layer * f_trunc;
          
          tau_layer   = (1.0 - omega_f) * cfg.delta_tau[l];
          omega_layer = omega_layer * (1.0 - f_trunc) / (1.0 - omega_f);
          
          std::vector<double> chi_star(two_M);
          double inv_1mf = 1.0 / (1.0 - f_trunc);
          
          for (int ll = 0; ll < two_M; ++ll)
            chi_star[ll] = (chi_full[ll] - f_trunc) * inv_1mf;
          
          const auto& Pl = ws->getLegendrePolynomials(two_M, mu);
          computePhaseMatricesFromLegendre(chi_star, Pl, wt, Ppp, Ppm);
          
          if (has_solar)
            computeSolarPhaseVectorsDynamic(chi_star, Pl, wt, cfg.solar_mu, p_plus_solar, p_minus_solar);
        }
        else 
        {
          std::vector<double> chi(chi_full.begin(),
              chi_full.begin() + std::min(static_cast<int>(chi_full.size()), two_M));
          const auto& Pl = ws->getLegendrePolynomials(
              static_cast<int>(chi.size()), mu);
          
          computePhaseMatricesFromLegendre(chi, Pl, wt, Ppp, Ppm);
          
          if (has_solar)
            computeSolarPhaseVectorsDynamic(chi, Pl, wt, cfg.solar_mu, p_plus_solar, p_minus_solar);
        }
      }
      else 
      {
        const auto& Pl = ws->getLegendrePolynomials(
            static_cast<int>(chi_full.size()), mu);
        
        computePhaseMatricesFromLegendre(chi_full, Pl, wt, Ppp, Ppm);
        
        if (has_solar)
          computeSolarPhaseVectorsDynamic(chi_full, Pl, wt, cfg.solar_mu, p_plus_solar, p_minus_solar);
      }
    }

    tau_used[l] = tau_layer;
    layer_rtj.push_back(
        dynDoubling(tau_layer, omega_layer, B_layer_top, B_layer_bot,
                     Ppp, Ppm, mu, wt,
                     cfg.solar_flux, cfg.solar_mu, tau_cumulative,
                     p_plus_solar, p_minus_solar));
    tau_cumulative += tau_layer;
  }

  int ltot = nlay;
  bool has_surface = !cfg.use_diffusion_lower_bc && (cfg.surface_albedo > 0.0 || B_surface > 0.0);

  if (has_surface) 
  {
    DynLayerMatrices surf(nmu);
    surf.T_ab = DynamicMatrix(nmu);
    surf.T_ba = DynamicMatrix(nmu);
    double A = cfg.surface_albedo;
    
    for (int i = 0; i < nmu; ++i) 
    {
      for (int j = 0; j < nmu; ++j) 
      {
        double r = 2.0 * A * mu[j] * wt[j] * xfac;
        surf.R_ab(i, j) = r;
        surf.R_ba(i, j) = r;
      }

      surf.s_up[i]   = (1.0 - A) * B_surface;
      surf.s_down[i] = 0.0;
      
      if (has_solar && A > 0.0)
        surf.s_up_solar[i] = (A / PI) * cfg.solar_flux * cfg.solar_mu
                             * std::exp(-tau_cumulative / cfg.solar_mu);
    }
    
    if (A > 0.0) surf.is_scattering = true;
    layer_rtj.push_back(std::move(surf));
    ltot++;
  }

  std::vector<DynLayerMatrices> rbase;
  rbase.reserve(ltot + 1);
  rbase.emplace_back(nmu);
  rbase.push_back(layer_rtj[ltot - 1]);
  
  for (int l = 1; l < ltot; ++l) 
  {
    int k = ltot - 1 - l;
    rbase.push_back(dynAddLayers(layer_rtj[k], rbase[l]));
  }

  std::vector<DynLayerMatrices> rtop;
  rtop.reserve(ltot + 1);
  rtop.emplace_back(nmu);
  rtop.push_back(layer_rtj[0]);
  
  for (int l = 1; l < ltot; ++l)
    rtop.push_back(dynAddLayers(rtop[l], layer_rtj[l]));

  std::vector<double> I_top_down(nmu, B_top_emission);
  std::vector<double> I_bot_up(nmu, 0.0);
  
  if (cfg.use_diffusion_lower_bc) 
  {
    double B_bottom = B[nlay];
    double dtau_last = tau_used[nlay - 1];
    double dB_dtau = (dtau_last > 0.0) ? (B_bottom - B[nlay - 1]) / dtau_last : 0.0;
    
    for (int i = 0; i < nmu; ++i)
      I_bot_up[i] = B_bottom + mu[i] * dB_dtau;
  }
  else if (!has_surface) 
  {
    for (int i = 0; i < nmu; ++i) I_bot_up[i] = B_surface;
  }

  int n_interfaces = nlay + 1;
  RTOutput result;
  result.flux_up.resize(n_interfaces, 0.0);
  result.flux_down.resize(n_interfaces, 0.0);
  result.mean_intensity.resize(n_interfaces, 0.0);
  result.flux_direct.resize(n_interfaces, 0.0);

  if (cfg.solar_flux > 0.0 && cfg.solar_mu > 0.0) 
  {
    double tau_cum = 0.0;
    result.flux_direct[0] = cfg.solar_flux * cfg.solar_mu;
    
    for (int l = 0; l < nlay; ++l) 
    {
      tau_cum += tau_used[l];
      result.flux_direct[l + 1] = cfg.solar_flux * cfg.solar_mu
                                  * std::exp(-tau_cum / cfg.solar_mu);
    }
  }

  {
    auto& full = rbase[ltot];
    auto Iup = full.R_ab.multiply(I_top_down);
    auto Tbot = full.T_ba.multiply(I_bot_up);

    for (int i = 0; i < nmu; ++i)
      Iup[i] += Tbot[i] + full.s_up[i] + full.s_up_solar[i];
    
    for (int i = 0; i < nmu; ++i) {
      result.flux_up[0]        += 2.0 * PI * wt[i] * mu[i] * Iup[i];
      result.flux_down[0]      += 2.0 * PI * wt[i] * mu[i] * I_top_down[i];
      result.mean_intensity[0] += 0.5 * wt[i] * (Iup[i] + I_top_down[i]);
    }
  }

  for (int l = 1; l <= nlay; ++l) 
  {
    int n_top = l;
    int n_base = ltot - l;
    std::vector<double> Iup, Idown;

    if (n_base > 0 && n_top > 0) 
    {
      auto I_id = DynamicMatrix::identity(nmu);
      // computeIup
      {
        DynamicMatrix to_inv = I_id.add(rbase[n_base].R_ab.multiply(rtop[n_top].R_ba), -1.0);
        auto t1 = rbase[n_base].T_ba.multiply(I_bot_up);
        auto t2_pre = rtop[n_top].T_ab.multiply(I_top_down);
        auto t2 = rbase[n_base].R_ab.multiply(t2_pre);
        std::vector<double> sd(nmu);
        
        for (int i = 0; i < nmu; ++i) 
          sd[i] = rtop[n_top].s_down[i] + rtop[n_top].s_down_solar[i];
        
        auto t3 = rbase[n_base].R_ab.multiply(sd);
        std::vector<double> rhs(nmu);
        
        for (int i = 0; i < nmu; ++i)
          rhs[i] = t1[i] + t2[i] + t3[i] + rbase[n_base].s_up[i] + rbase[n_base].s_up_solar[i];
        
        Iup = to_inv.solve(rhs);
      }
      // computeIdown
      {
        DynamicMatrix to_inv = I_id.add(rtop[n_top].R_ba.multiply(rbase[n_base].R_ab), -1.0);
        auto t1 = rtop[n_top].T_ab.multiply(I_top_down);
        auto t2_pre = rbase[n_base].T_ba.multiply(I_bot_up);
        auto t2 = rtop[n_top].R_ba.multiply(t2_pre);
        std::vector<double> su(nmu);
        
        for (int i = 0; i < nmu; ++i) 
          su[i] = rbase[n_base].s_up[i] + rbase[n_base].s_up_solar[i];
        
        auto t3 = rtop[n_top].R_ba.multiply(su);
        std::vector<double> rhs(nmu);
        
        for (int i = 0; i < nmu; ++i)
          rhs[i] = t1[i] + t2[i] + t3[i] + rtop[n_top].s_down[i] + rtop[n_top].s_down_solar[i];
        
        Idown = to_inv.solve(rhs);
      }
    }
    else if (n_base == 0) 
    {
      Idown = rtop[n_top].T_ab.multiply(I_top_down);
      auto RtopIbot = rtop[n_top].R_ba.multiply(I_bot_up);

      for (int i = 0; i < nmu; ++i)
        Idown[i] += RtopIbot[i] + rtop[n_top].s_down[i] + rtop[n_top].s_down_solar[i];
      
      Iup = I_bot_up;
    }
    else 
    {
      Iup.assign(nmu, 0.0);
      Idown = I_top_down;
    }

    for (int i = 0; i < nmu; ++i) 
    {
      result.flux_up[l]        += 2.0 * PI * wt[i] * mu[i] * Iup[i];
      result.flux_down[l]      += 2.0 * PI * wt[i] * mu[i] * Idown[i];
      result.mean_intensity[l] += 0.5 * wt[i] * (Iup[i] + Idown[i]);
    }
  }

  if (cfg.index_from_bottom) 
  {
    std::reverse(result.flux_up.begin(), result.flux_up.end());
    std::reverse(result.flux_down.begin(), result.flux_down.end());
    std::reverse(result.mean_intensity.begin(), result.mean_intensity.end());
    std::reverse(result.flux_direct.begin(), result.flux_direct.end());
  }

  return result;
}


// ============================================================================
//  Public solver: runtime dispatch to templated implementation
// ============================================================================

static RTOutput solveDispatch(const ADConfig& config, SolverWorkspace* ws) 
{
  switch (config.num_quadrature) {
    case 2:  return solveImpl<2>(config, ws);
    case 4:  return solveImpl<4>(config, ws);
    case 8:  return solveImpl<8>(config, ws);
    case 16: return solveImpl<16>(config, ws);
    case 32: return solveImpl<32>(config, ws);
    default: return solveDynamic(config, ws);
  }
}


RTOutput solve(const ADConfig& config) 
{
  return solveDispatch(config, nullptr);
}


RTOutput solve(const ADConfig& config, SolverWorkspace& workspace) 
{
  return solveDispatch(config, &workspace);
}

} // namespace adrt
