/// @file cuda_batched_solver.cuh
/// @brief Batched cuBLAS solver for N>=16, processing all wavenumbers in parallel.
///
/// Uses cublasSgemmStridedBatched for matrix multiplies and
/// cublasDgetrfBatched/cublasDgetrsBatched for double-precision LU solves.
/// Row-major storage with cuBLAS transpose trick eliminates explicit transposes.

#pragma once

#include "cuda_solver.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace adrt {
namespace cuda {

/// Launch the batched cuBLAS solver for N >= 16.
/// Same interface as the templated path — called from solveBatch().
void solveBatchedCublas(
    const BatchConfig& config,
    const DeviceData& data,
    cudaStream_t stream);

} // namespace cuda
} // namespace adrt
