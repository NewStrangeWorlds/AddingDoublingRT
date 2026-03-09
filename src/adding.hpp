/// @file adding.h
/// @brief Adding algorithm: combine two layers into a composite layer.

#pragma once

#include "layer.hpp"
#include "matrix.hpp"

namespace adrt {

// ============================================================================
//  Source combination helper
// ============================================================================

template<int N>
void addSources(
    LayerMatrices<N>& ans,
    const LayerMatrices<N>& top,
    const LayerMatrices<N>& bot,
    const Matrix<N>& T_ba_D1,
    const Matrix<N>& T_bc_D2)
{
  using Vec = typename Matrix<N>::EigenVec;

  auto do_sources = [&](
    const Vec& s1_up,
    const Vec& s1_down,
    const Vec& s2_up,
    const Vec& s2_down,
    Vec& ans_up,
    Vec& ans_down)
  {
    Vec Rbc_s1m = bot.R_ab.multiply(s1_down);
    Vec rhs = s2_up + Rbc_s1m;
    ans_up = s1_up + T_ba_D1.multiply(rhs);

    Vec Rba_s2p = top.R_ba.multiply(s2_up);
    rhs = s1_down + Rba_s2p;
    ans_down = s2_down + T_bc_D2.multiply(rhs);
  };

  do_sources(top.s_up, top.s_down, bot.s_up, bot.s_down,
             ans.s_up, ans.s_down);
  do_sources(top.s_up_solar, top.s_down_solar, bot.s_up_solar, bot.s_down_solar,
             ans.s_up_solar, ans.s_down_solar);
}


// ============================================================================
//  General adding (both layers scattering)
// ============================================================================

template<int N>
LayerMatrices<N> addLayersGeneral(
    const LayerMatrices<N>& top,
    const LayerMatrices<N>& bot)
{
  LayerMatrices<N> ans;
  ans.is_scattering = true;

  auto I = Matrix<N>::identity();

  Matrix<N> A1 = I.add(bot.R_ab.multiply(top.R_ba), -1.0);
  Matrix<N> A2 = I.add(top.R_ba.multiply(bot.R_ab), -1.0);

  Matrix<N> T_ba_D1 = A1.rightSolveMatrix(top.T_ba);
  Matrix<N> T_bc_D2 = A2.rightSolveMatrix(bot.T_ab);

  Matrix<N> temp1 = T_ba_D1.multiply(bot.R_ab);
  ans.R_ab = top.R_ab.add(temp1.multiply(top.T_ab));
  Matrix<N> temp2 = T_bc_D2.multiply(top.R_ba);
  ans.R_ba = bot.R_ba.add(temp2.multiply(bot.T_ba));

  ans.T_ab = T_bc_D2.multiply(top.T_ab);
  ans.T_ba = T_ba_D1.multiply(bot.T_ba);

  addSources(ans, top, bot, T_ba_D1, T_bc_D2);

  return ans;
}


// ============================================================================
//  Non-scattering top layer optimisation
// ============================================================================

template<int N>
LayerMatrices<N> addLayersNonscatTop(
    const LayerMatrices<N>& top,
    const LayerMatrices<N>& bot)
{
  using Vec = typename Matrix<N>::EigenVec;
  LayerMatrices<N> ans;
  ans.is_scattering = bot.is_scattering;

  Vec t;

  for (int i = 0; i < N; ++i)
    t[i] = top.T_ab(i, i);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ans.R_ab(i, j) = t[i] * bot.R_ab(i, j) * t[j];

  ans.R_ba = bot.R_ba;

  ans.T_ab = Matrix<N>();
  
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ans.T_ab(i, j) = bot.T_ab(i, j) * t[j];

  ans.T_ba = Matrix<N>();

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ans.T_ba(i, j) = t[i] * bot.T_ba(i, j);

  auto do_sources = [&](
    const Vec& s1_up,
    const Vec& s1_down,
    const Vec& s2_up,
    const Vec& s2_down,
    Vec& ans_up,
    Vec& ans_down)
  {
    Vec Rbc_s1m = bot.R_ab.multiply(s1_down);
    for (int i = 0; i < N; ++i)
      ans_up[i] = s1_up[i] + t[i] * (s2_up[i] + Rbc_s1m[i]);

    Vec Tbc_s1m = bot.T_ab.multiply(s1_down);
    ans_down = s2_down + Tbc_s1m;
  };

  do_sources(top.s_up, top.s_down, bot.s_up, bot.s_down,
             ans.s_up, ans.s_down);
  do_sources(top.s_up_solar, top.s_down_solar, bot.s_up_solar, bot.s_down_solar,
             ans.s_up_solar, ans.s_down_solar);

  return ans;
}


// ============================================================================
//  Non-scattering bottom layer optimisation
// ============================================================================

template<int N>
LayerMatrices<N> addLayersNonscatBot(
    const LayerMatrices<N>& top,
    const LayerMatrices<N>& bot)
{
  using Vec = typename Matrix<N>::EigenVec;
  LayerMatrices<N> ans;
  ans.is_scattering = top.is_scattering;

  Vec t;
  
  for (int i = 0; i < N; ++i)
    t[i] = bot.T_ab(i, i);

  ans.R_ab = top.R_ab;

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ans.R_ba(i, j) = t[i] * top.R_ba(i, j) * t[j];

  ans.T_ab = Matrix<N>();
  
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ans.T_ab(i, j) = t[i] * top.T_ab(i, j);

  ans.T_ba = Matrix<N>();
  
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ans.T_ba(i, j) = top.T_ba(i, j) * t[j];

  auto do_sources = [&](
      const Vec& s1_up,
      const Vec& s1_down,
      const Vec& s2_up,
      const Vec& s2_down,
      Vec& ans_up,
      Vec& ans_down)
  {
    Vec Tba_s2p = top.T_ba.multiply(s2_up);
    ans_up = s1_up + Tba_s2p;

    Vec Rba_s2p = top.R_ba.multiply(s2_up);
    
    for (int i = 0; i < N; ++i)
      ans_down[i] = s2_down[i] + t[i] * (s1_down[i] + Rba_s2p[i]);
  };

  do_sources(top.s_up, top.s_down, bot.s_up, bot.s_down,
             ans.s_up, ans.s_down);
  do_sources(top.s_up_solar, top.s_down_solar, bot.s_up_solar, bot.s_down_solar,
             ans.s_up_solar, ans.s_down_solar);

  return ans;
}


// ============================================================================
//  Dispatch: choose optimal adding variant
// ============================================================================

template<int N>
LayerMatrices<N> addLayers(
    const LayerMatrices<N>& top,
    const LayerMatrices<N>& bot)
{
  if (!top.is_scattering)
    return addLayersNonscatTop<N>(top, bot);
  if (!bot.is_scattering)
    return addLayersNonscatBot<N>(top, bot);
  
  return addLayersGeneral<N>(top, bot);
}

} // namespace adrt
