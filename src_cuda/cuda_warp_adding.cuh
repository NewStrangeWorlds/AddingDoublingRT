/// @file cuda_warp_adding.cuh
/// @brief Warp-cooperative adding algorithm for the CUDA adding-doubling solver.
///
/// Distributed version of cuda_adding.cuh: N threads cooperate, each owning
/// one row of each matrix and one element of each vector.

#pragma once

#include "cuda_warp_layer.cuh"
#include "cuda_warp_matrix.cuh"

namespace adrt {
namespace cuda {

// ============================================================================
//  Source combination helper
// ============================================================================

template<int N>
__device__ __forceinline__ void warp_add_sources(
    WarpLayerMatrices<N>& ans,
    const WarpLayerMatrices<N>& top,
    const WarpLayerMatrices<N>& bot,
    const WarpRow<N>& T_ba_D1,
    const WarpRow<N>& T_bc_D2,
    int row_id)
{
  // --- Thermal ---
  {
    float Rbc_s1m = wmat_vec_multiply<N>(bot.R_ab, top.s_down);
    float rhs = bot.s_up + Rbc_s1m;
    float contrib = wmat_vec_multiply<N>(T_ba_D1, rhs);
    ans.s_up = top.s_up + contrib;

    float Rba_s2p = wmat_vec_multiply<N>(top.R_ba, bot.s_up);
    rhs = top.s_down + Rba_s2p;
    contrib = wmat_vec_multiply<N>(T_bc_D2, rhs);
    ans.s_down = bot.s_down + contrib;
  }

  // --- Solar ---
  {
    float Rbc_s1m = wmat_vec_multiply<N>(bot.R_ab, top.s_down_solar);
    float rhs = bot.s_up_solar + Rbc_s1m;
    float contrib = wmat_vec_multiply<N>(T_ba_D1, rhs);
    ans.s_up_solar = top.s_up_solar + contrib;

    float Rba_s2p = wmat_vec_multiply<N>(top.R_ba, bot.s_up_solar);
    rhs = top.s_down_solar + Rba_s2p;
    contrib = wmat_vec_multiply<N>(T_bc_D2, rhs);
    ans.s_down_solar = bot.s_down_solar + contrib;
  }
}


// ============================================================================
//  General adding (both layers scattering)
// ============================================================================

template<int N>
__device__ __forceinline__ void warp_add_layers_general(
    WarpLayerMatrices<N>& ans,
    const WarpLayerMatrices<N>& top,
    const WarpLayerMatrices<N>& bot,
    int row_id)
{
  ans.is_scattering = true;

  // A1 = I - R_bot * R_top_ba
  WarpRow<N> RbotRtop, A1;
  wmat_multiply<N>(RbotRtop, bot.R_ab, top.R_ba);
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float delta_ij = (row_id == j) ? 1.0f : 0.0f;
    A1[j] = delta_ij - RbotRtop[j];
  }

  // A2 = I - R_top_ba * R_bot
  WarpRow<N> RtopRbot, A2;
  wmat_multiply<N>(RtopRbot, top.R_ba, bot.R_ab);
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float delta_ij = (row_id == j) ? 1.0f : 0.0f;
    A2[j] = delta_ij - RtopRbot[j];
  }

  // T_ba_D1 = T_top_ba * (I - R_bot * R_top_ba)^{-1}
  // solve: T_ba_D1 * A1 = T_top_ba
  WarpRow<N> T_ba_D1;
  wmat_right_solve_matrix<N>(T_ba_D1, A1, top.T_ba, row_id);

  // T_bc_D2 = T_bot_ab * (I - R_top_ba * R_bot)^{-1}
  WarpRow<N> T_bc_D2;
  wmat_right_solve_matrix<N>(T_bc_D2, A2, bot.T_ab, row_id);

  // Composite reflection
  WarpRow<N> temp1, temp1_T;
  wmat_multiply<N>(temp1, T_ba_D1, bot.R_ab);
  wmat_multiply<N>(temp1_T, temp1, top.T_ab);
  wmat_add<N>(ans.R_ab, top.R_ab, temp1_T, 1.0f);

  WarpRow<N> temp2, temp2_T;
  wmat_multiply<N>(temp2, T_bc_D2, top.R_ba);
  wmat_multiply<N>(temp2_T, temp2, bot.T_ba);
  wmat_add<N>(ans.R_ba, bot.R_ba, temp2_T, 1.0f);

  // Composite transmission
  wmat_multiply<N>(ans.T_ab, T_bc_D2, top.T_ab);
  wmat_multiply<N>(ans.T_ba, T_ba_D1, bot.T_ba);

  // Source combination
  warp_add_sources<N>(ans, top, bot, T_ba_D1, T_bc_D2, row_id);
}


// ============================================================================
//  Non-scattering top layer optimisation
// ============================================================================

template<int N>
__device__ __forceinline__ void warp_add_layers_nonscat_top(
    WarpLayerMatrices<N>& ans,
    const WarpLayerMatrices<N>& top,
    const WarpLayerMatrices<N>& bot,
    int row_id)
{
  ans.is_scattering = bot.is_scattering;

  // Extract diagonal transmission: t = T_ab(row_id, row_id)
  float t = top.T_ab[row_id];  // diagonal element of this thread's row

  // R_ab = diag(t) * bot.R_ab * diag(t)
  // Row i: ans.R_ab(i,j) = t[i] * bot.R_ab(i,j) * t[j]
  // t[i] is local (= t for this thread). t[j] needs shuffle.
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float tj = __shfl_sync(warp_group_mask<N>(), t, j, N);
    ans.R_ab[j] = t * bot.R_ab[j] * tj;
  }

  wmat_copy<N>(ans.R_ba, bot.R_ba);

  // T_ab: bot.T_ab * diag(t)  → row i: ans(i,j) = bot.T_ab(i,j) * t[j]
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float tj = __shfl_sync(warp_group_mask<N>(), t, j, N);
    ans.T_ab[j] = bot.T_ab[j] * tj;
  }

  // T_ba: diag(t) * bot.T_ba  → row i: ans(i,j) = t[i] * bot.T_ba(i,j)
  #pragma unroll
  for (int j = 0; j < N; ++j)
    ans.T_ba[j] = t * bot.T_ba[j];

  // Source combination (simplified — no matrix inversion needed)
  // Thermal
  {
    float Rbc_s1m = wmat_vec_multiply<N>(bot.R_ab, top.s_down);
    ans.s_up = top.s_up + t * (bot.s_up + Rbc_s1m);

    float Tbc_s1m = wmat_vec_multiply<N>(bot.T_ab, top.s_down);
    ans.s_down = bot.s_down + Tbc_s1m;
  }
  // Solar
  {
    float Rbc_s1m = wmat_vec_multiply<N>(bot.R_ab, top.s_down_solar);
    ans.s_up_solar = top.s_up_solar + t * (bot.s_up_solar + Rbc_s1m);

    float Tbc_s1m = wmat_vec_multiply<N>(bot.T_ab, top.s_down_solar);
    ans.s_down_solar = bot.s_down_solar + Tbc_s1m;
  }
}


// ============================================================================
//  Non-scattering bottom layer optimisation
// ============================================================================

template<int N>
__device__ __forceinline__ void warp_add_layers_nonscat_bot(
    WarpLayerMatrices<N>& ans,
    const WarpLayerMatrices<N>& top,
    const WarpLayerMatrices<N>& bot,
    int row_id)
{
  ans.is_scattering = top.is_scattering;

  float t = bot.T_ab[row_id];  // diagonal element

  wmat_copy<N>(ans.R_ab, top.R_ab);

  // R_ba = diag(t) * top.R_ba * diag(t)
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float tj = __shfl_sync(warp_group_mask<N>(), t, j, N);
    ans.R_ba[j] = t * top.R_ba[j] * tj;
  }

  // T_ab: diag(t) * top.T_ab
  #pragma unroll
  for (int j = 0; j < N; ++j)
    ans.T_ab[j] = t * top.T_ab[j];

  // T_ba: top.T_ba * diag(t)
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    float tj = __shfl_sync(warp_group_mask<N>(), t, j, N);
    ans.T_ba[j] = top.T_ba[j] * tj;
  }

  // Source combination (simplified)
  // Thermal
  {
    float Tba_s2p = wmat_vec_multiply<N>(top.T_ba, bot.s_up);
    ans.s_up = top.s_up + Tba_s2p;

    float Rba_s2p = wmat_vec_multiply<N>(top.R_ba, bot.s_up);
    ans.s_down = bot.s_down + t * (top.s_down + Rba_s2p);
  }
  // Solar
  {
    float Tba_s2p = wmat_vec_multiply<N>(top.T_ba, bot.s_up_solar);
    ans.s_up_solar = top.s_up_solar + Tba_s2p;

    float Rba_s2p = wmat_vec_multiply<N>(top.R_ba, bot.s_up_solar);
    ans.s_down_solar = bot.s_down_solar + t * (top.s_down_solar + Rba_s2p);
  }
}


// ============================================================================
//  Dispatch: choose optimal adding variant
// ============================================================================

template<int N>
__device__ __forceinline__ void warp_add_layers(
    WarpLayerMatrices<N>& ans,
    const WarpLayerMatrices<N>& top,
    const WarpLayerMatrices<N>& bot,
    int row_id)
{
  if (!top.is_scattering)
    warp_add_layers_nonscat_top<N>(ans, top, bot, row_id);
  else if (!bot.is_scattering)
    warp_add_layers_nonscat_bot<N>(ans, top, bot, row_id);
  else
    warp_add_layers_general<N>(ans, top, bot, row_id);
}

} // namespace cuda
} // namespace adrt
