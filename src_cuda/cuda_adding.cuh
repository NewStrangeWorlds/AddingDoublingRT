/// @file cuda_adding.cuh
/// @brief Device-side adding algorithm for the CUDA adding-doubling solver.
///
/// Combines two adjacent layer systems into a composite system.
/// Direct port of adding.hpp.

#pragma once

#include "cuda_layer.cuh"
#include "cuda_matrix.cuh"

namespace adrt {
namespace cuda {

// ============================================================================
//  Source combination helper
// ============================================================================

template<int N>
__device__ __forceinline__ void add_sources(
    GpuLayerMatrices<N>& ans,
    const GpuLayerMatrices<N>& top,
    const GpuLayerMatrices<N>& bot,
    const GpuMatrix<N>& T_ba_D1,
    const GpuMatrix<N>& T_bc_D2)
{
  // Process thermal and solar sources via the same logic
  // Lambda-like approach: do each pair (thermal, solar) explicitly.

  // --- Thermal ---
  {
    GpuVec<N> Rbc_s1m;
    mat_vec_multiply<N>(Rbc_s1m, bot.R_ab, top.s_down);

    GpuVec<N> rhs;
    vec_add<N>(rhs, bot.s_up, Rbc_s1m);

    GpuVec<N> contrib;
    mat_vec_multiply<N>(contrib, T_ba_D1, rhs);
    vec_add<N>(ans.s_up, top.s_up, contrib);

    GpuVec<N> Rba_s2p;
    mat_vec_multiply<N>(Rba_s2p, top.R_ba, bot.s_up);
    vec_add<N>(rhs, top.s_down, Rba_s2p);
    mat_vec_multiply<N>(contrib, T_bc_D2, rhs);
    vec_add<N>(ans.s_down, bot.s_down, contrib);
  }

  // --- Solar ---
  {
    GpuVec<N> Rbc_s1m;
    mat_vec_multiply<N>(Rbc_s1m, bot.R_ab, top.s_down_solar);

    GpuVec<N> rhs;
    vec_add<N>(rhs, bot.s_up_solar, Rbc_s1m);

    GpuVec<N> contrib;
    mat_vec_multiply<N>(contrib, T_ba_D1, rhs);
    vec_add<N>(ans.s_up_solar, top.s_up_solar, contrib);

    GpuVec<N> Rba_s2p;
    mat_vec_multiply<N>(Rba_s2p, top.R_ba, bot.s_up_solar);
    vec_add<N>(rhs, top.s_down_solar, Rba_s2p);
    mat_vec_multiply<N>(contrib, T_bc_D2, rhs);
    vec_add<N>(ans.s_down_solar, bot.s_down_solar, contrib);
  }
}


// ============================================================================
//  General adding (both layers scattering)
// ============================================================================

template<int N>
__device__ __forceinline__ void add_layers_general(
    GpuLayerMatrices<N>& ans,
    const GpuLayerMatrices<N>& top,
    const GpuLayerMatrices<N>& bot)
{
  ans.is_scattering = true;

  GpuMatrix<N> I_mat;
  mat_set_identity<N>(I_mat);

  // A1 = I - R_bot * R_top_ba
  GpuMatrix<N> RbotRtop, A1;
  mat_multiply<N>(RbotRtop, bot.R_ab, top.R_ba);
  mat_add<N>(A1, I_mat, RbotRtop, -1.0);

  // A2 = I - R_top_ba * R_bot
  GpuMatrix<N> RtopRbot, A2;
  mat_multiply<N>(RtopRbot, top.R_ba, bot.R_ab);
  mat_add<N>(A2, I_mat, RtopRbot, -1.0);

  // T_ba_D1 = T_top_ba * (I - R_bot * R_top_ba)^{-1}
  // i.e. solve: T_ba_D1 * A1 = T_top_ba
  GpuMatrix<N> T_ba_D1;
  mat_right_solve_matrix<N>(T_ba_D1, A1, top.T_ba);

  // T_bc_D2 = T_bot_ab * (I - R_top_ba * R_bot)^{-1}
  GpuMatrix<N> T_bc_D2;
  mat_right_solve_matrix<N>(T_bc_D2, A2, bot.T_ab);

  // Composite reflection
  GpuMatrix<N> temp1, temp1_T;
  mat_multiply<N>(temp1, T_ba_D1, bot.R_ab);
  mat_multiply<N>(temp1_T, temp1, top.T_ab);
  mat_add<N>(ans.R_ab, top.R_ab, temp1_T, 1.0);

  GpuMatrix<N> temp2, temp2_T;
  mat_multiply<N>(temp2, T_bc_D2, top.R_ba);
  mat_multiply<N>(temp2_T, temp2, bot.T_ba);
  mat_add<N>(ans.R_ba, bot.R_ba, temp2_T, 1.0);

  // Composite transmission
  mat_multiply<N>(ans.T_ab, T_bc_D2, top.T_ab);
  mat_multiply<N>(ans.T_ba, T_ba_D1, bot.T_ba);

  // Source combination
  add_sources<N>(ans, top, bot, T_ba_D1, T_bc_D2);
}


// ============================================================================
//  Non-scattering top layer optimisation
// ============================================================================

template<int N>
__device__ __forceinline__ void add_layers_nonscat_top(
    GpuLayerMatrices<N>& ans,
    const GpuLayerMatrices<N>& top,
    const GpuLayerMatrices<N>& bot)
{
  ans.is_scattering = bot.is_scattering;

  // Extract diagonal transmission of top layer
  GpuVec<N> t;
  #pragma unroll
  for (int i = 0; i < N; ++i)
    t[i] = top.T_ab(i, i);

  // R_ab = diag(t) * bot.R_ab * diag(t)
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      ans.R_ab(i, j) = t[i] * bot.R_ab(i, j) * t[j];

  mat_copy<N>(ans.R_ba, bot.R_ba);

  // T_ab: bot.T_ab * diag(t)
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      ans.T_ab(i, j) = bot.T_ab(i, j) * t[j];

  // T_ba: diag(t) * bot.T_ba
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      ans.T_ba(i, j) = t[i] * bot.T_ba(i, j);

  // Source combination (simplified — no matrix inversion needed)
  // Thermal
  {
    GpuVec<N> Rbc_s1m;
    mat_vec_multiply<N>(Rbc_s1m, bot.R_ab, top.s_down);
    #pragma unroll
    for (int i = 0; i < N; ++i)
      ans.s_up[i] = top.s_up[i] + t[i] * (bot.s_up[i] + Rbc_s1m[i]);

    GpuVec<N> Tbc_s1m;
    mat_vec_multiply<N>(Tbc_s1m, bot.T_ab, top.s_down);
    vec_add<N>(ans.s_down, bot.s_down, Tbc_s1m);
  }
  // Solar
  {
    GpuVec<N> Rbc_s1m;
    mat_vec_multiply<N>(Rbc_s1m, bot.R_ab, top.s_down_solar);
    #pragma unroll
    for (int i = 0; i < N; ++i)
      ans.s_up_solar[i] = top.s_up_solar[i] + t[i] * (bot.s_up_solar[i] + Rbc_s1m[i]);

    GpuVec<N> Tbc_s1m;
    mat_vec_multiply<N>(Tbc_s1m, bot.T_ab, top.s_down_solar);
    vec_add<N>(ans.s_down_solar, bot.s_down_solar, Tbc_s1m);
  }
}


// ============================================================================
//  Non-scattering bottom layer optimisation
// ============================================================================

template<int N>
__device__ __forceinline__ void add_layers_nonscat_bot(
    GpuLayerMatrices<N>& ans,
    const GpuLayerMatrices<N>& top,
    const GpuLayerMatrices<N>& bot)
{
  ans.is_scattering = top.is_scattering;

  GpuVec<N> t;
  #pragma unroll
  for (int i = 0; i < N; ++i)
    t[i] = bot.T_ab(i, i);

  mat_copy<N>(ans.R_ab, top.R_ab);

  // R_ba = diag(t) * top.R_ba * diag(t)
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      ans.R_ba(i, j) = t[i] * top.R_ba(i, j) * t[j];

  // T_ab: diag(t) * top.T_ab
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      ans.T_ab(i, j) = t[i] * top.T_ab(i, j);

  // T_ba: top.T_ba * diag(t)
  #pragma unroll
  for (int i = 0; i < N; ++i)
    #pragma unroll
    for (int j = 0; j < N; ++j)
      ans.T_ba(i, j) = top.T_ba(i, j) * t[j];

  // Source combination (simplified)
  // Thermal
  {
    GpuVec<N> Tba_s2p;
    mat_vec_multiply<N>(Tba_s2p, top.T_ba, bot.s_up);
    vec_add<N>(ans.s_up, top.s_up, Tba_s2p);

    GpuVec<N> Rba_s2p;
    mat_vec_multiply<N>(Rba_s2p, top.R_ba, bot.s_up);
    #pragma unroll
    for (int i = 0; i < N; ++i)
      ans.s_down[i] = bot.s_down[i] + t[i] * (top.s_down[i] + Rba_s2p[i]);
  }
  // Solar
  {
    GpuVec<N> Tba_s2p;
    mat_vec_multiply<N>(Tba_s2p, top.T_ba, bot.s_up_solar);
    vec_add<N>(ans.s_up_solar, top.s_up_solar, Tba_s2p);

    GpuVec<N> Rba_s2p;
    mat_vec_multiply<N>(Rba_s2p, top.R_ba, bot.s_up_solar);
    #pragma unroll
    for (int i = 0; i < N; ++i)
      ans.s_down_solar[i] = bot.s_down_solar[i] + t[i] * (top.s_down_solar[i] + Rba_s2p[i]);
  }
}


// ============================================================================
//  Dispatch: choose optimal adding variant
// ============================================================================

template<int N>
__device__ __forceinline__ void add_layers(
    GpuLayerMatrices<N>& ans,
    const GpuLayerMatrices<N>& top,
    const GpuLayerMatrices<N>& bot)
{
  if (!top.is_scattering)
    add_layers_nonscat_top<N>(ans, top, bot);
  else if (!bot.is_scattering)
    add_layers_nonscat_bot<N>(ans, top, bot);
  else
    add_layers_general<N>(ans, top, bot);
}

} // namespace cuda
} // namespace adrt
