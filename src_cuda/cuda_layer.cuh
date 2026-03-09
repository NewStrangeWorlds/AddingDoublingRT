/// @file cuda_layer.cuh
/// @brief Device-side layer data structures for the CUDA adding-doubling solver.

#pragma once

#include "cuda_matrix.cuh"

namespace adrt {
namespace cuda {

/// Per-layer reflection/transmission matrices and source vectors (device-side).
template<int N>
struct GpuLayerMatrices {
  GpuMatrix<N> R_ab;   ///< Reflection (above → below)
  GpuMatrix<N> R_ba;   ///< Reflection (below → above)
  GpuMatrix<N> T_ab;   ///< Transmission (above → below)
  GpuMatrix<N> T_ba;   ///< Transmission (below → above)

  GpuVec<N> s_up;          ///< Thermal source (upward)
  GpuVec<N> s_down;        ///< Thermal source (downward)
  GpuVec<N> s_up_solar;    ///< Solar source (upward)
  GpuVec<N> s_down_solar;  ///< Solar source (downward)

  bool is_scattering;

  /// Initialise to a transparent layer (identity transmission, zero everything else).
  __device__ __forceinline__ void set_transparent() {
    mat_set_zero<N>(R_ab);
    mat_set_zero<N>(R_ba);
    mat_set_identity<N>(T_ab);
    mat_set_identity<N>(T_ba);
    vec_set_zero<N>(s_up);
    vec_set_zero<N>(s_down);
    vec_set_zero<N>(s_up_solar);
    vec_set_zero<N>(s_down_solar);
    is_scattering = false;
  }
};

} // namespace cuda
} // namespace adrt
