/// @file cuda_warp_layer.cuh
/// @brief Warp-cooperative layer data structures for the CUDA adding-doubling solver.
///
/// Distributed version of GpuLayerMatrices: each thread holds one row of each
/// matrix and one element of each vector.

#pragma once

#include "cuda_warp_matrix.cuh"

namespace adrt {
namespace cuda {

/// Per-layer reflection/transmission matrices and source vectors (distributed).
/// Thread with row_id=i owns row i of each matrix and element i of each vector.
template<int N>
struct WarpLayerMatrices {
  WarpRow<N> R_ab;   ///< Reflection (above → below), row row_id
  WarpRow<N> R_ba;   ///< Reflection (below → above), row row_id
  WarpRow<N> T_ab;   ///< Transmission (above → below), row row_id
  WarpRow<N> T_ba;   ///< Transmission (below → above), row row_id

  float s_up;          ///< Thermal source (upward), element row_id
  float s_down;        ///< Thermal source (downward), element row_id
  float s_up_solar;    ///< Solar source (upward), element row_id
  float s_down_solar;  ///< Solar source (downward), element row_id

  bool is_scattering;

  /// Initialise to a transparent layer.
  __device__ __forceinline__ void set_transparent(int row_id) {
    wmat_set_zero<N>(R_ab);
    wmat_set_zero<N>(R_ba);
    wmat_set_identity<N>(T_ab, row_id);
    wmat_set_identity<N>(T_ba, row_id);
    s_up = 0.0f;
    s_down = 0.0f;
    s_up_solar = 0.0f;
    s_down_solar = 0.0f;
    is_scattering = false;
  }
};

} // namespace cuda
} // namespace adrt
