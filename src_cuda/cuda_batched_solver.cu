/// @file cuda_batched_solver.cu
/// @brief Batched cuBLAS implementation of the adding-doubling solver for N>=16.
///
/// Processes all wavenumbers simultaneously using:
/// - cublasSgemmStridedBatched for N×N matrix multiplies
/// - cublasSgetrfBatched/cublasSgetrsBatched for single-precision LU solves (doubling)
/// - cublasDgetrfBatched/cublasDgetrsBatched for double-precision LU solves (adding)
/// - Custom kernels for element-wise operations

#include "cuda_batched_solver.cuh"
#include "cuda_batched_kernels.cuh"
#include "cuda_quadrature.cuh"
#include "cuda_planck.cuh"

#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace adrt {
namespace cuda {

// ============================================================================
//  Workspace: all device memory for the batched solver
// ============================================================================

struct BatchedWorkspace {
  int N, nwav, nlay;
  cublasHandle_t handle;

  // Layer state arrays: each is [nwav * N*N] for matrices, [nwav * N] for vectors
  // We keep two layer structs: "composite" (accumulated) and "current" (single layer)
  // Plus a "combined" temp for adding output

  // composite layer (rbase)
  float *comp_R_ab, *comp_R_ba, *comp_T_ab, *comp_T_ba;
  float *comp_s_up, *comp_s_down, *comp_s_up_solar, *comp_s_down_solar;

  // current layer (from doubling)
  float *cur_R_ab, *cur_R_ba, *cur_T_ab, *cur_T_ba;
  float *cur_s_up, *cur_s_down, *cur_s_up_solar, *cur_s_down_solar;

  // combined layer (adding output)
  float *comb_R_ab, *comb_R_ba, *comb_T_ab, *comb_T_ba;
  float *comb_s_up, *comb_s_down, *comb_s_up_solar, *comb_s_down_solar;

  // Doubling temporaries
  float *Gpp, *Gpm;           // [nwav * N*N]
  float *R_k, *T_k;           // [nwav * N*N]
  float *tempA, *tempB;       // [nwav * N*N] scratch matrices
  float *tempC;               // [nwav * N*N] extra scratch

  // Source vectors for doubling
  float *y_k, *z_k;           // [nwav * N]
  float *y_k2, *z_k2;         // [nwav * N] swap buffers
  float *s_up_sol, *s_down_sol; // [nwav * N]
  float *g_k, *gamma_sol;     // [nwav]
  float *tau0, *half_tau0_sq;  // [nwav]

  // Per-wavenumber scalars for current layer
  float *tau_scaled, *omega_scaled; // [nwav]
  float *B_bar, *B_d;         // [nwav]

  // Adding temporaries
  float *add_T_ba_D1, *add_T_bc_D2; // [nwav * N*N]

  // Vector temporaries for source combination
  float *vec_tmp1, *vec_tmp2, *vec_tmp3, *vec_tmp4; // [nwav * N]

  // Single-precision LU workspace (for doubling — well-conditioned)
  float **fA_ptrs;             // [nwav]
  float **fB_ptrs;             // [nwav]
  float *fA_lu;                // [nwav * N*N] scratch for LU (A gets destroyed)
  int *d_pivot;                // [nwav * N]
  int *d_info;                 // [nwav]

  // Double-precision LU workspace (for adding — can be ill-conditioned)
  double *dA;                  // [nwav * N*N]
  double *dB;                  // [nwav * N*N]
  double **dA_ptrs;            // [nwav]
  double **dB_ptrs;            // [nwav]
  int *d_pivot_dp;             // [nwav * N]
  int *d_info_dp;              // [nwav]

  // Planck levels
  float *B_levels;             // [nwav * nlev]

  // Precomputed phase matrices
  float *phase_Ppp, *phase_Ppm; // [nlay * N*N]
  float *phase_f_trunc;       // [nlay]
  float *phase_solar_pp, *phase_solar_pm; // [nlay * N] or nullptr

  // Quadrature data (device copies for runtime-N kernels)
  float *d_mu_rt, *d_wt_rt;   // [N]
  float *d_Pl_rt;             // [nmom * N]
  float d_xfac_rt;

  // Boundary intensities
  float *I_top_down, *I_bot_up; // [nwav * N]

  // Total optical depth per wavenumber
  float *tau_total; // [nwav]

  // Cumulative optical depth above current layer (per wavenumber, for solar attenuation)
  float *tau_above; // [nwav]

  // Single big allocation
  void* workspace_mem;
  size_t workspace_size;
};

static void allocateWorkspace(BatchedWorkspace& ws, int N, int nwav, int nlay, int nlev,
                              bool has_solar, int nmom) {
  ws.N = N;
  ws.nwav = nwav;
  ws.nlay = nlay;
  ws.workspace_mem = nullptr;

  size_t mat_bytes = (size_t)nwav * N * N * sizeof(float);
  size_t vec_bytes = (size_t)nwav * N * sizeof(float);
  size_t dmat_bytes = (size_t)nwav * N * N * sizeof(double);

  auto mf = [](float** p, size_t bytes) { cudaMalloc(p, bytes); cudaMemset(*p, 0, bytes); };
  auto md = [](double** p, size_t bytes) { cudaMalloc(p, bytes); };
  auto mi = [](int** p, size_t bytes) { cudaMalloc(p, bytes); };

  // Composite layer
  mf(&ws.comp_R_ab, mat_bytes); mf(&ws.comp_R_ba, mat_bytes);
  mf(&ws.comp_T_ab, mat_bytes); mf(&ws.comp_T_ba, mat_bytes);
  mf(&ws.comp_s_up, vec_bytes); mf(&ws.comp_s_down, vec_bytes);
  mf(&ws.comp_s_up_solar, vec_bytes); mf(&ws.comp_s_down_solar, vec_bytes);

  // Current layer
  mf(&ws.cur_R_ab, mat_bytes); mf(&ws.cur_R_ba, mat_bytes);
  mf(&ws.cur_T_ab, mat_bytes); mf(&ws.cur_T_ba, mat_bytes);
  mf(&ws.cur_s_up, vec_bytes); mf(&ws.cur_s_down, vec_bytes);
  mf(&ws.cur_s_up_solar, vec_bytes); mf(&ws.cur_s_down_solar, vec_bytes);

  // Combined layer
  mf(&ws.comb_R_ab, mat_bytes); mf(&ws.comb_R_ba, mat_bytes);
  mf(&ws.comb_T_ab, mat_bytes); mf(&ws.comb_T_ba, mat_bytes);
  mf(&ws.comb_s_up, vec_bytes); mf(&ws.comb_s_down, vec_bytes);
  mf(&ws.comb_s_up_solar, vec_bytes); mf(&ws.comb_s_down_solar, vec_bytes);

  // Doubling matrices
  mf(&ws.Gpp, mat_bytes); mf(&ws.Gpm, mat_bytes);
  mf(&ws.R_k, mat_bytes); mf(&ws.T_k, mat_bytes);
  mf(&ws.tempA, mat_bytes); mf(&ws.tempB, mat_bytes);
  mf(&ws.tempC, mat_bytes);

  // Doubling vectors
  mf(&ws.y_k, vec_bytes); mf(&ws.z_k, vec_bytes);
  mf(&ws.y_k2, vec_bytes); mf(&ws.z_k2, vec_bytes);
  mf(&ws.s_up_sol, vec_bytes); mf(&ws.s_down_sol, vec_bytes);

  // Per-wav scalars
  mf(&ws.g_k, nwav * sizeof(float)); mf(&ws.gamma_sol, nwav * sizeof(float));
  mf(&ws.tau0, nwav * sizeof(float)); mf(&ws.half_tau0_sq, nwav * sizeof(float));
  mf(&ws.tau_scaled, nwav * sizeof(float)); mf(&ws.omega_scaled, nwav * sizeof(float));
  mf(&ws.B_bar, nwav * sizeof(float)); mf(&ws.B_d, nwav * sizeof(float));

  // Adding
  mf(&ws.add_T_ba_D1, mat_bytes); mf(&ws.add_T_bc_D2, mat_bytes);

  // Vec temps
  mf(&ws.vec_tmp1, vec_bytes); mf(&ws.vec_tmp2, vec_bytes);
  mf(&ws.vec_tmp3, vec_bytes); mf(&ws.vec_tmp4, vec_bytes);

  // Boundary
  mf(&ws.I_top_down, vec_bytes); mf(&ws.I_bot_up, vec_bytes);

  // tau_total and tau_above
  mf(&ws.tau_total, nwav * sizeof(float));
  mf(&ws.tau_above, nwav * sizeof(float));

  // B_levels
  mf(&ws.B_levels, (size_t)nwav * nlev * sizeof(float));

  // Phase
  mf(&ws.phase_Ppp, (size_t)nlay * N * N * sizeof(float));
  mf(&ws.phase_Ppm, (size_t)nlay * N * N * sizeof(float));
  mf(&ws.phase_f_trunc, nlay * sizeof(float));
  if (has_solar) {
    mf(&ws.phase_solar_pp, (size_t)nlay * N * sizeof(float));
    mf(&ws.phase_solar_pm, (size_t)nlay * N * sizeof(float));
  } else {
    ws.phase_solar_pp = nullptr;
    ws.phase_solar_pm = nullptr;
  }

  // Quadrature
  mf(&ws.d_mu_rt, N * sizeof(float));
  mf(&ws.d_wt_rt, N * sizeof(float));
  mf(&ws.d_Pl_rt, (size_t)nmom * N * sizeof(float));

  // Single-precision LU workspace (doubling)
  mf(&ws.fA_lu, mat_bytes);
  cudaMalloc(&ws.fA_ptrs, nwav * sizeof(float*));
  cudaMalloc(&ws.fB_ptrs, nwav * sizeof(float*));
  mi(&ws.d_pivot, (size_t)nwav * N * sizeof(int));
  mi(&ws.d_info, nwav * sizeof(int));

  // Double-precision LU workspace (adding)
  md(&ws.dA, dmat_bytes);
  md(&ws.dB, dmat_bytes);
  cudaMalloc(&ws.dA_ptrs, nwav * sizeof(double*));
  cudaMalloc(&ws.dB_ptrs, nwav * sizeof(double*));
  mi(&ws.d_pivot_dp, (size_t)nwav * N * sizeof(int));
  mi(&ws.d_info_dp, nwav * sizeof(int));
}

static void freeWorkspace(BatchedWorkspace& ws) {
  auto cf = [](float* p) { if (p) cudaFree(p); };
  auto cd = [](double* p) { if (p) cudaFree(p); };

  cf(ws.comp_R_ab); cf(ws.comp_R_ba); cf(ws.comp_T_ab); cf(ws.comp_T_ba);
  cf(ws.comp_s_up); cf(ws.comp_s_down); cf(ws.comp_s_up_solar); cf(ws.comp_s_down_solar);
  cf(ws.cur_R_ab); cf(ws.cur_R_ba); cf(ws.cur_T_ab); cf(ws.cur_T_ba);
  cf(ws.cur_s_up); cf(ws.cur_s_down); cf(ws.cur_s_up_solar); cf(ws.cur_s_down_solar);
  cf(ws.comb_R_ab); cf(ws.comb_R_ba); cf(ws.comb_T_ab); cf(ws.comb_T_ba);
  cf(ws.comb_s_up); cf(ws.comb_s_down); cf(ws.comb_s_up_solar); cf(ws.comb_s_down_solar);
  cf(ws.Gpp); cf(ws.Gpm); cf(ws.R_k); cf(ws.T_k);
  cf(ws.tempA); cf(ws.tempB); cf(ws.tempC);
  cf(ws.y_k); cf(ws.z_k); cf(ws.y_k2); cf(ws.z_k2);
  cf(ws.s_up_sol); cf(ws.s_down_sol);
  cf(ws.g_k); cf(ws.gamma_sol); cf(ws.tau0); cf(ws.half_tau0_sq);
  cf(ws.tau_scaled); cf(ws.omega_scaled); cf(ws.B_bar); cf(ws.B_d);
  cf(ws.add_T_ba_D1); cf(ws.add_T_bc_D2);
  cf(ws.vec_tmp1); cf(ws.vec_tmp2); cf(ws.vec_tmp3); cf(ws.vec_tmp4);
  cf(ws.I_top_down); cf(ws.I_bot_up);
  cf(ws.tau_total); cf(ws.tau_above); cf(ws.B_levels);
  cf(ws.phase_Ppp); cf(ws.phase_Ppm); cf(ws.phase_f_trunc);
  cf(ws.phase_solar_pp); cf(ws.phase_solar_pm);
  cf(ws.d_mu_rt); cf(ws.d_wt_rt); cf(ws.d_Pl_rt);
  cf(ws.fA_lu);
  if (ws.fA_ptrs) cudaFree(ws.fA_ptrs);
  if (ws.fB_ptrs) cudaFree(ws.fB_ptrs);
  if (ws.d_pivot) cudaFree(ws.d_pivot);
  if (ws.d_info) cudaFree(ws.d_info);
  cd(ws.dA); cd(ws.dB);
  if (ws.dA_ptrs) cudaFree(ws.dA_ptrs);
  if (ws.dB_ptrs) cudaFree(ws.dB_ptrs);
  if (ws.d_pivot_dp) cudaFree(ws.d_pivot_dp);
  if (ws.d_info_dp) cudaFree(ws.d_info_dp);
}


// ============================================================================
//  cuBLAS helpers
// ============================================================================

/// Row-major C = A * B via cuBLAS: gemm(B^T, A^T) => (AB)^T in col-major = AB in row-major
static void batchedGemm(cublasHandle_t handle, int N, int nwav,
                        const float* A, const float* B, float* C,
                        float alpha = 1.0f, float beta = 0.0f,
                        cudaStream_t stream = 0)
{
  long long strideA = N * N;
  long long strideB = N * N;
  long long strideC = N * N;

  // C_rm = A_rm * B_rm  =>  cublas: C_cm = B_cm * A_cm
  // In row-major: A_rm stored, cuBLAS reads as A^T_cm
  // So cublasSgemm(B_ptr, A_ptr) computes B^T * A^T = (AB)^T in col-major = AB in row-major
  cublasSgemmStridedBatched(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, N, N,
      &alpha,
      B, N, strideB,  // "A" in cuBLAS = our B
      A, N, strideA,  // "B" in cuBLAS = our A
      &beta,
      C, N, strideC,
      nwav);
}

/// Batched matrix-vector multiply: y = A * x (row-major)
/// Treats x as [nwav * N * 1] and uses gemmStridedBatched with M=N, N=1, K=N
static void batchedGemv(cublasHandle_t handle, int N, int nwav,
                        const float* A, const float* x, float* y,
                        float alpha = 1.0f, float beta = 0.0f)
{
  // Row-major A*x: cuBLAS sees A^T (col-major), so we need A^T * x in col-major
  // But A^T * x computes the wrong thing. Instead:
  // y = A * x (row-major) = (x^T * A^T)^T
  // Use gemm with x as 1×N and A^T as N×N: result is 1×N = y^T
  // Actually simpler: just use cublasSgemvStridedBatched or treat as gemm with n=1.
  //
  // For row-major y = A * x:
  //   cuBLAS col-major: y = A^T_col * x doesn't work (gives A^T * x)
  //   Instead: y_i = sum_j A(i,j) * x(j) = sum_j A_rm[i*N+j] * x[j]
  //   cuBLAS col-major sees A_rm as N×N matrix where col j = row j of A_rm
  //   So col-major matrix M has M(i,j) = A_rm(j,i) = A(j,i) = A^T(i,j)
  //   cuBLAS computes M*x = A^T*x, but we want A*x
  //   Solution: use CUBLAS_OP_T: cublasSgemv(CUBLAS_OP_T, N, N, alpha, A_rm, N, x, 1, beta, y, 1)
  //   This computes M^T * x = A * x. Correct!
  //
  // Use strided batched gemm with n_cols=1 for the batched case.

  long long strideA = N * N;
  long long strideX = N;
  long long strideY = N;

  // y = A * x (row-major) => cuBLAS: y = op(M) * x where M = A stored row-major
  // M in col-major = A^T, so op = CUBLAS_OP_T gives A^T^T = A
  // Using gemm: treat as m=N, n=1, k=N
  // cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, N, 1, N, alpha, A, N, x, N, beta, y, N)
  cublasSgemmStridedBatched(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      N, 1, N,
      &alpha,
      A, N, strideA,
      x, N, strideX,
      &beta,
      y, N, strideY,
      nwav);
}


/// Single-precision batched LU solve: solve X * A = B for X (right-solve, transpose_A=false)
/// or A * X = B for X (left-solve, transpose_A=true).
/// A is [nwav * N * N] float (row-major). If A_destructible, A is used in-place (destroyed by getrf).
/// B is [nwav * N * N] float (row-major), X overwrites B in-place.
///
/// Row-major trick:
///   Right-solve X*A=B => A^T * X^T = B^T => cuBLAS getrf(A_rm) + getrs(CUBLAS_OP_N)
///   Left-solve  A*X=B => cuBLAS getrf(A_rm) + getrs(CUBLAS_OP_T)
static void batchedLUSolveMatrix(BatchedWorkspace& ws, cudaStream_t stream,
                                 float* A_float, float* B_float_inout,
                                 bool transpose_A = false,
                                 bool A_destructible = false)
{
  int N = ws.N;
  int nwav = ws.nwav;

  using namespace batched;

  float* A_lu = A_float;
  if (!A_destructible) {
    int mat_count = nwav * N * N;
    batchedMatCopyKernel<<<divUp(mat_count, BLOCK), BLOCK, 0, stream>>>(
        mat_count, A_float, ws.fA_lu);
    A_lu = ws.fA_lu;
  }

  // Build pointer arrays
  batchedBuildFloatPointerArrayKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
      nwav, N * N, A_lu, ws.fA_ptrs);
  batchedBuildFloatPointerArrayKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
      nwav, N * N, B_float_inout, ws.fB_ptrs);

  // LU factorize
  cublasSgetrfBatched(ws.handle, N, ws.fA_ptrs, N, ws.d_pivot, ws.d_info, nwav);

  // Solve
  cublasOperation_t op = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  int h_info = 0;
  cublasSgetrsBatched(ws.handle, op, N, N,
                      (const float**)ws.fA_ptrs, N, ws.d_pivot,
                      ws.fB_ptrs, N, &h_info, nwav);
}

/// Double-precision batched LU solve for adding (can be ill-conditioned for ω→1).
/// Same interface as batchedLUSolveMatrix but converts to double internally.
static void batchedLUSolveMatrixDouble(BatchedWorkspace& ws, cudaStream_t stream,
                                       float* A_float, float* B_float_inout,
                                       bool transpose_A = false)
{
  int N = ws.N;
  int nwav = ws.nwav;
  int mat_count = nwav * N * N;

  using namespace batched;

  // Float -> Double
  batchedFloat2DoubleKernel<<<divUp(mat_count, BLOCK), BLOCK, 0, stream>>>(
      mat_count, A_float, ws.dA);
  batchedFloat2DoubleKernel<<<divUp(mat_count, BLOCK), BLOCK, 0, stream>>>(
      mat_count, B_float_inout, ws.dB);

  // Build pointer arrays
  batchedBuildPointerArrayKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
      nwav, N * N, ws.dA, ws.dA_ptrs);
  batchedBuildPointerArrayKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
      nwav, N * N, ws.dB, ws.dB_ptrs);

  // LU factorize
  cublasDgetrfBatched(ws.handle, N, ws.dA_ptrs, N, ws.d_pivot_dp, ws.d_info_dp, nwav);

  // Solve
  cublasOperation_t op = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  int h_info = 0;
  cublasDgetrsBatched(ws.handle, op, N, N,
                      (const double**)ws.dA_ptrs, N, ws.d_pivot_dp,
                      ws.dB_ptrs, N, &h_info, nwav);

  // Double -> Float
  batchedDouble2FloatKernel<<<divUp(mat_count, BLOCK), BLOCK, 0, stream>>>(
      mat_count, ws.dB, B_float_inout);
}


// ============================================================================
//  Stride-extract kernel (extract every stride-th element)
// ============================================================================

__global__ void strideExtractKernel(
    int n, int stride, int offset,
    const float* __restrict__ src,
    float* __restrict__ dst)
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= n) return;
  dst[w] = src[w * stride + offset];
}


// ============================================================================
//  Doubling for one layer across all wavenumbers
// ============================================================================

/// Doubling for one layer across all wavenumbers.
/// Expects ws.tau_scaled and ws.omega_scaled to already be extracted and delta-M scaled.
static void batchedDoubling(BatchedWorkspace& ws, cudaStream_t stream,
                            int layer_idx,
                            float solar_flux, float solar_mu,
                            const float* tau_above,
                            bool has_solar, int nn_max)
{
  int N = ws.N;
  int nwav = ws.nwav;
  size_t mat_total = (size_t)nwav * N * N;
  size_t vec_total = (size_t)nwav * N;

  using namespace batched;

  // Build Gpp, Gpm
  batchedBuildGppGpmKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N,
      ws.phase_Ppp + (size_t)layer_idx * N * N,
      ws.phase_Ppm + (size_t)layer_idx * N * N,
      ws.omega_scaled,
      ws.d_mu_rt, ws.d_wt_rt,
      ws.Gpp, ws.Gpm);

  // Compute tau0 and half_tau0_sq
  batchedComputeTau0Kernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
      nwav, ws.tau_scaled, ws.omega_scaled, nn_max,
      ws.tau0, ws.half_tau0_sq);

  // First-order thin-layer init: T_k = I - tau0*Gpp, R_k = tau0*Gpm
  batchedFirstOrderInitKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.tau0, ws.Gpp, ws.Gpm, ws.T_k, ws.R_k);

  // Second-order corrections via cuBLAS
  // T_k += half_tau0_sq * Gpp²
  batchedGemm(ws.handle, N, nwav, ws.Gpp, ws.Gpp, ws.tempA, 1.0f, 0.0f, stream);
  batchedAddScaledMatrixKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.half_tau0_sq, 1.0f, ws.tempA, ws.T_k);

  // T_k += half_tau0_sq * Gpm²
  batchedGemm(ws.handle, N, nwav, ws.Gpm, ws.Gpm, ws.tempA, 1.0f, 0.0f, stream);
  batchedAddScaledMatrixKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.half_tau0_sq, 1.0f, ws.tempA, ws.T_k);

  // R_k -= half_tau0_sq * Gpp*Gpm
  batchedGemm(ws.handle, N, nwav, ws.Gpp, ws.Gpm, ws.tempA, 1.0f, 0.0f, stream);
  batchedAddScaledMatrixKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.half_tau0_sq, -1.0f, ws.tempA, ws.R_k);

  // R_k -= half_tau0_sq * Gpm*Gpp
  batchedGemm(ws.handle, N, nwav, ws.Gpm, ws.Gpp, ws.tempA, 1.0f, 0.0f, stream);
  batchedAddScaledMatrixKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.half_tau0_sq, -1.0f, ws.tempA, ws.R_k);

  // Source init
  batchedSourceInitKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.tau0, ws.omega_scaled, ws.d_mu_rt,
      solar_flux, solar_mu, tau_above,
      has_solar,
      has_solar ? ws.phase_solar_pp + (size_t)layer_idx * N : nullptr,
      has_solar ? ws.phase_solar_pm + (size_t)layer_idx * N : nullptr,
      ws.y_k, ws.z_k, ws.s_up_sol, ws.s_down_sol,
      ws.g_k, ws.gamma_sol);

  // --- Doubling iterations ---
  for (int k = 0; k < nn_max; ++k) {
    // tempA = R_k * R_k
    batchedGemm(ws.handle, N, nwav, ws.R_k, ws.R_k, ws.tempA, 1.0f, 0.0f, stream);

    // tempA = I - R²
    batchedNegateAddIdentityKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
        nwav, N, ws.tempA);

    // tempB = T_k (copy for right-solve input)
    batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
        mat_total, ws.T_k, ws.tempB);

    // Right-solve: tempB * tempA = T_k  =>  tempB = T_k * (I-R²)^{-1} = TG
    // tempA is not needed after this, so allow destructive use
    batchedLUSolveMatrix(ws, stream, ws.tempA, ws.tempB, false, true);
    // Now tempB = TG

    // tempA = TG * R_k = TGR
    batchedGemm(ws.handle, N, nwav, ws.tempB, ws.R_k, ws.tempA, 1.0f, 0.0f, stream);
    // Now tempA = TGR, tempB = TG

    // --- Thermal source update ---
    // zpgy = z_k + g_k * y_k
    batchedComputeZpgyKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
        nwav, N, ws.z_k, ws.y_k, ws.g_k, ws.vec_tmp1); // vec_tmp1 = zpgy

    // TG_zpgy = TG * zpgy (matrix-vector multiply)
    batchedGemv(ws.handle, N, nwav, ws.tempB, ws.vec_tmp1, ws.vec_tmp2, 1.0f, 0.0f);
    // TGR_zpgy = TGR * zpgy
    batchedGemv(ws.handle, N, nwav, ws.tempA, ws.vec_tmp1, ws.vec_tmp3, 1.0f, 0.0f);
    // TG_y = TG * y_k
    batchedGemv(ws.handle, N, nwav, ws.tempB, ws.y_k, ws.vec_tmp4, 1.0f, 0.0f);
    // TGR_y = TGR * y_k  (reuse vec_tmp1 which held zpgy, now dead)
    batchedGemv(ws.handle, N, nwav, ws.tempA, ws.y_k, ws.vec_tmp1, 1.0f, 0.0f);

    // Combine: y_new, z_new
    batchedThermalSourceCombineKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
        nwav, N,
        ws.vec_tmp2, ws.vec_tmp3, ws.vec_tmp4, ws.vec_tmp1,
        ws.y_k, ws.z_k, ws.g_k,
        ws.y_k2, ws.z_k2);

    // Solar source update
    if (has_solar) {
      // R_sdown = R_k * s_down_sol
      batchedGemv(ws.handle, N, nwav, ws.R_k, ws.s_down_sol, ws.vec_tmp2, 1.0f, 0.0f);
      // R_sup = R_k * s_up_sol
      batchedGemv(ws.handle, N, nwav, ws.R_k, ws.s_up_sol, ws.vec_tmp3, 1.0f, 0.0f);

      // Build rhs
      batchedSolarRhsKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
          nwav, N,
          ws.vec_tmp2, ws.vec_tmp3,
          ws.s_up_sol, ws.s_down_sol, ws.gamma_sol,
          ws.vec_tmp1, ws.vec_tmp4); // rhs_up, rhs_down

      // TG * rhs_up
      batchedGemv(ws.handle, N, nwav, ws.tempB, ws.vec_tmp1, ws.vec_tmp2, 1.0f, 0.0f);
      // TG * rhs_down
      batchedGemv(ws.handle, N, nwav, ws.tempB, ws.vec_tmp4, ws.vec_tmp3, 1.0f, 0.0f);

      // Combine solar
      batchedSolarCombineKernel2<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
          nwav, N, ws.vec_tmp2, ws.vec_tmp3, ws.gamma_sol,
          ws.s_up_sol, ws.s_down_sol);

      // gamma_sol = gamma_sol^2
      batchedSquareGammaSolKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
          nwav, ws.gamma_sol);
    }

    // Update R_k: R_k += TGR * T_k
    batchedGemm(ws.handle, N, nwav, ws.tempA, ws.T_k, ws.R_k, 1.0f, 1.0f, stream);

    // Update T_k: T_k = TG * T_k => compute into tempC, then copy
    batchedGemm(ws.handle, N, nwav, ws.tempB, ws.T_k, ws.tempC, 1.0f, 0.0f, stream);
    batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
        mat_total, ws.tempC, ws.T_k);

    // Swap y/z buffers
    std::swap(ws.y_k, ws.y_k2);
    std::swap(ws.z_k, ws.z_k2);

    // g_k *= 2
    batchedDoubleGkKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(nwav, ws.g_k);
  }

  // Assemble into current layer
  batchedAssembleLayerKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.R_k, ws.T_k,
      ws.cur_R_ab, ws.cur_R_ba, ws.cur_T_ab, ws.cur_T_ba);

  batchedAssembleSourceKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.y_k, ws.z_k, ws.s_up_sol, ws.s_down_sol,
      ws.B_bar, ws.B_d,
      ws.cur_s_up, ws.cur_s_down, ws.cur_s_up_solar, ws.cur_s_down_solar);
}


// ============================================================================
//  Adding: combine top (cur) and bot (comp) into comb
// ============================================================================

static void batchedAdding(BatchedWorkspace& ws, cudaStream_t stream)
{
  int N = ws.N;
  int nwav = ws.nwav;
  size_t mat_total = (size_t)nwav * N * N;
  size_t vec_total = (size_t)nwav * N;

  using namespace batched;

  // General adding: both layers may be scattering
  // top = cur (current layer), bot = comp (accumulated composite)

  // Step 1: tempA = bot.R_ab * top.R_ba
  batchedGemm(ws.handle, N, nwav, ws.comp_R_ab, ws.cur_R_ba, ws.tempA, 1.0f, 0.0f, stream);

  // tempA = I - tempA
  batchedNegateAddIdentityKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.tempA);

  // T_ba_D1 = top.T_ba * tempA^{-1}  (right solve: T_ba_D1 * tempA = top.T_ba)
  // Use double-precision LU for adding — (I - R*R) can be ill-conditioned for ω→1
  batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      mat_total, ws.cur_T_ba, ws.add_T_ba_D1);
  batchedLUSolveMatrixDouble(ws, stream, ws.tempA, ws.add_T_ba_D1, false);

  // Step 2: tempA = top.R_ba * bot.R_ab
  batchedGemm(ws.handle, N, nwav, ws.cur_R_ba, ws.comp_R_ab, ws.tempA, 1.0f, 0.0f, stream);
  batchedNegateAddIdentityKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.tempA);

  // T_bc_D2 = bot.T_ab * tempA^{-1}
  batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      mat_total, ws.comp_T_ab, ws.add_T_bc_D2);
  batchedLUSolveMatrixDouble(ws, stream, ws.tempA, ws.add_T_bc_D2, false);

  // Composite R_ab = top.R_ab + T_ba_D1 * bot.R_ab * top.T_ab
  batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      mat_total, ws.cur_R_ab, ws.comb_R_ab);
  batchedGemm(ws.handle, N, nwav, ws.add_T_ba_D1, ws.comp_R_ab, ws.tempA, 1.0f, 0.0f, stream);
  batchedGemm(ws.handle, N, nwav, ws.tempA, ws.cur_T_ab, ws.comb_R_ab, 1.0f, 1.0f, stream);

  // Composite R_ba = bot.R_ba + T_bc_D2 * top.R_ba * bot.T_ba
  batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
      mat_total, ws.comp_R_ba, ws.comb_R_ba);
  batchedGemm(ws.handle, N, nwav, ws.add_T_bc_D2, ws.cur_R_ba, ws.tempA, 1.0f, 0.0f, stream);
  batchedGemm(ws.handle, N, nwav, ws.tempA, ws.comp_T_ba, ws.comb_R_ba, 1.0f, 1.0f, stream);

  // Composite T_ab = T_bc_D2 * top.T_ab
  batchedGemm(ws.handle, N, nwav, ws.add_T_bc_D2, ws.cur_T_ab, ws.comb_T_ab, 1.0f, 0.0f, stream);

  // Composite T_ba = T_ba_D1 * bot.T_ba
  batchedGemm(ws.handle, N, nwav, ws.add_T_ba_D1, ws.comp_T_ba, ws.comb_T_ba, 1.0f, 0.0f, stream);

  // --- Source combination ---
  // Thermal upward: s_up = top.s_up + T_ba_D1 * (bot.s_up + bot.R_ab * top.s_down)
  batchedGemv(ws.handle, N, nwav, ws.comp_R_ab, ws.cur_s_down, ws.vec_tmp1, 1.0f, 0.0f);
  batchedVecAddKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      vec_total, ws.comp_s_up, ws.vec_tmp1, ws.vec_tmp2);
  batchedGemv(ws.handle, N, nwav, ws.add_T_ba_D1, ws.vec_tmp2, ws.vec_tmp1, 1.0f, 0.0f);
  batchedVecAddKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      vec_total, ws.cur_s_up, ws.vec_tmp1, ws.comb_s_up);

  // Thermal downward: s_down = bot.s_down + T_bc_D2 * (top.s_down + top.R_ba * bot.s_up)
  batchedGemv(ws.handle, N, nwav, ws.cur_R_ba, ws.comp_s_up, ws.vec_tmp1, 1.0f, 0.0f);
  batchedVecAddKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      vec_total, ws.cur_s_down, ws.vec_tmp1, ws.vec_tmp2);
  batchedGemv(ws.handle, N, nwav, ws.add_T_bc_D2, ws.vec_tmp2, ws.vec_tmp1, 1.0f, 0.0f);
  batchedVecAddKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      vec_total, ws.comp_s_down, ws.vec_tmp1, ws.comb_s_down);

  // Solar upward: same pattern with solar vectors
  batchedGemv(ws.handle, N, nwav, ws.comp_R_ab, ws.cur_s_down_solar, ws.vec_tmp1, 1.0f, 0.0f);
  batchedVecAddKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      vec_total, ws.comp_s_up_solar, ws.vec_tmp1, ws.vec_tmp2);
  batchedGemv(ws.handle, N, nwav, ws.add_T_ba_D1, ws.vec_tmp2, ws.vec_tmp1, 1.0f, 0.0f);
  batchedVecAddKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      vec_total, ws.cur_s_up_solar, ws.vec_tmp1, ws.comb_s_up_solar);

  // Solar downward
  batchedGemv(ws.handle, N, nwav, ws.cur_R_ba, ws.comp_s_up_solar, ws.vec_tmp1, 1.0f, 0.0f);
  batchedVecAddKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      vec_total, ws.cur_s_down_solar, ws.vec_tmp1, ws.vec_tmp2);
  batchedGemv(ws.handle, N, nwav, ws.add_T_bc_D2, ws.vec_tmp2, ws.vec_tmp1, 1.0f, 0.0f);
  batchedVecAddKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      vec_total, ws.comp_s_down_solar, ws.vec_tmp1, ws.comb_s_down_solar);

  // Swap combined <-> composite pointers (avoid 8 copy kernels)
  std::swap(ws.comp_R_ab, ws.comb_R_ab);
  std::swap(ws.comp_R_ba, ws.comb_R_ba);
  std::swap(ws.comp_T_ab, ws.comb_T_ab);
  std::swap(ws.comp_T_ba, ws.comb_T_ba);
  std::swap(ws.comp_s_up, ws.comb_s_up);
  std::swap(ws.comp_s_down, ws.comb_s_down);
  std::swap(ws.comp_s_up_solar, ws.comb_s_up_solar);
  std::swap(ws.comp_s_down_solar, ws.comb_s_down_solar);
}


// ============================================================================
//  Compute B_bar and B_d for a layer
// ============================================================================

__global__ void batchedComputeBBarBdKernel(
    int nwav, int nlay, int layer_idx,
    const float* __restrict__ B_levels,  // [nwav * nlev]
    const float* __restrict__ tau,       // [nwav]
    float* __restrict__ B_bar,           // [nwav]
    float* __restrict__ B_d)             // [nwav]
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;
  int nlev = nlay + 1;
  float B_top = B_levels[w * nlev + layer_idx];
  float B_bot = B_levels[w * nlev + layer_idx + 1];
  B_bar[w] = (B_bot + B_top) * 0.5f;
  float t = tau[w];
  B_d[w] = (t > 0.0f) ? (B_bot - B_top) / t : 0.0f;
}


// ============================================================================
//  Compute total optical depth per wavenumber
// ============================================================================

__global__ void batchedComputeTauTotalKernel(
    int nwav, int nlay,
    const float* __restrict__ delta_tau,  // [nwav * nlay]
    float* __restrict__ tau_total)        // [nwav]
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwav) return;
  float sum = 0.0f;
  for (int l = 0; l < nlay; ++l)
    sum += delta_tau[w * nlay + l];
  tau_total[w] = sum;
}


// ============================================================================
//  Check if layer needs scattering (host-side helper)
// ============================================================================

/// Compute nn_max for a specific layer by reading omega values from device
static int computeLayerNnMax(const float* d_omega_layer, int nwav,
                              const float* d_tau_layer, cudaStream_t stream)
{
  // Copy to host for analysis
  std::vector<float> omega_host(nwav), tau_host(nwav);
  cudaMemcpyAsync(omega_host.data(), d_omega_layer, nwav * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(tau_host.data(), d_tau_layer, nwav * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  return batched::computeNnMax(tau_host, omega_host, nwav);
}


// ============================================================================
//  Main entry point
// ============================================================================

void solveBatchedCublas(
    const BatchConfig& config,
    const DeviceData& data,
    cudaStream_t stream)
{
  int N = config.num_quadrature;
  int nwav = config.num_wavenumbers;
  int nlay = config.num_layers;
  int nlev = nlay + 1;
  int nmom = config.num_moments_max;
  bool has_solar = (config.solar_flux > 0.0 && config.solar_mu > 0.0);
  float solar_flux = static_cast<float>(config.solar_flux);
  float solar_mu = static_cast<float>(config.solar_mu);

  using namespace batched;

  // Upload quadrature data to constant memory (for Planck kernel)
  uploadQuadratureData(N, nmom);

  // Allocate workspace
  BatchedWorkspace ws;
  allocateWorkspace(ws, N, nwav, nlay, nlev, has_solar, nmom);
  cublasCreate(&ws.handle);
  cublasSetStream(ws.handle, stream);

  // Upload quadrature to runtime-accessible device memory
  {
    std::vector<double> mu_d, wt_d;
    hostGaussLegendre(N, mu_d, wt_d);

    std::vector<float> mu_f(N), wt_f(N);
    for (int i = 0; i < N; ++i) {
      mu_f[i] = static_cast<float>(mu_d[i]);
      wt_f[i] = static_cast<float>(wt_d[i]);
    }

    double xfac_sum = 0.0;
    for (int i = 0; i < N; ++i) xfac_sum += mu_d[i] * wt_d[i];
    ws.d_xfac_rt = static_cast<float>(0.5 / xfac_sum);

    cudaMemcpyAsync(ws.d_mu_rt, mu_f.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.d_wt_rt, wt_f.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Legendre polynomials: Pl[l*N+i]
    if (nmom > 0) {
      auto Pl = hostPrecomputeLegendrePolynomials(nmom, mu_d);
      std::vector<float> Pl_f(nmom * N);
      for (int l = 0; l < nmom; ++l)
        for (int i = 0; i < N; ++i)
          Pl_f[l * N + i] = static_cast<float>(Pl[l * N + i]);
      cudaMemcpyAsync(ws.d_Pl_rt, Pl_f.data(), nmom * N * sizeof(float),
                      cudaMemcpyHostToDevice, stream);
    }
  }

  // Precompute phase matrices (shared moments only — for now always precompute)
  if (data.phase_moments_shared) {
    batchedPhaseMatrixKernel<<<divUp(nlay, 64), 64, 0, stream>>>(
        nlay, N, nmom, config.use_delta_m,
        solar_mu, has_solar,
        data.phase_moments,
        ws.d_mu_rt, ws.d_wt_rt, ws.d_Pl_rt,
        ws.phase_Ppp, ws.phase_Ppm, ws.phase_f_trunc,
        ws.phase_solar_pp, ws.phase_solar_pm);
  }
  // TODO: handle per-wavenumber phase moments if needed

  // Compute Planck levels
  batchedPlanckKernel<<<divUp(nwav * nlev, BLOCK), BLOCK, 0, stream>>>(
      nwav, nlev, config.use_thermal_emission,
      config.wavenumber_low, config.wavenumber_high,
      data.temperature, data.planck_levels, ws.B_levels);

  // Compute total optical depth
  batchedComputeTauTotalKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
      nwav, nlay, data.delta_tau, ws.tau_total);

  // --- Surface layer ---
  bool has_surface = !config.use_diffusion_lower_bc
                     && (config.surface_albedo > 0.0 || config.surface_emission > 0.0
                         || config.use_thermal_emission);
  bool comp_initialized = false;

  if (has_surface) {
    size_t mat_total = (size_t)nwav * N * N;
    size_t vec_total = (size_t)nwav * N;

    batchedSurfaceLayerKernel<<<divUp(std::max(mat_total, vec_total), BLOCK), BLOCK, 0, stream>>>(
        nwav, N,
        static_cast<float>(config.surface_albedo),
        static_cast<float>(config.surface_emission),
        solar_flux, solar_mu, has_solar,
        config.use_thermal_emission,
        ws.d_mu_rt, ws.d_wt_rt, ws.d_xfac_rt,
        data.surface_emission, ws.tau_total,
        ws.B_levels, nlev,
        ws.comp_R_ab, ws.comp_R_ba, ws.comp_T_ab, ws.comp_T_ba,
        ws.comp_s_up, ws.comp_s_down,
        ws.comp_s_up_solar, ws.comp_s_down_solar);
    comp_initialized = true;
  }

  // Precompute uniform nn_max across all layers (single host sync)
  int nn_max_global;
  {
    std::vector<float> all_tau(nwav * nlay), all_omega(nwav * nlay);
    cudaMemcpyAsync(all_tau.data(), data.delta_tau, nwav * nlay * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(all_omega.data(), data.single_scat_albedo, nwav * nlay * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    nn_max_global = 0;
    for (int l = 0; l < nlay; ++l) {
      // Extract per-wavenumber tau and omega for this layer
      std::vector<float> tau_l(nwav), omega_l(nwav);
      for (int w = 0; w < nwav; ++w) {
        tau_l[w] = all_tau[w * nlay + l];
        omega_l[w] = all_omega[w * nlay + l];
      }
      int nn = batched::computeNnMax(tau_l, omega_l, nwav);
      nn_max_global = std::max(nn_max_global, nn);
    }
  }

  // Initialize tau_above = tau_total (start from bottom, working up)
  {
    size_t sc_bytes = nwav * sizeof(float);
    cudaMemcpyAsync(ws.tau_above, ws.tau_total, sc_bytes, cudaMemcpyDeviceToDevice, stream);
  }

  // --- Process layers bottom to top ---
  for (int l = nlay - 1; l >= 0; --l) {
    // Extract tau and omega for this layer
    strideExtractKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
        nwav, nlay, l, data.delta_tau, ws.tau_scaled);
    strideExtractKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
        nwav, nlay, l, data.single_scat_albedo, ws.omega_scaled);

    // tau_above -= tau_layer (now tau_above = cumulative tau above layer l)
    batchedSubtractTauKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
        nwav, ws.tau_scaled, ws.tau_above);

    // Apply delta-M scaling
    if (config.use_delta_m) {
      batchedDeltaMScaleKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
          nwav,
          ws.phase_f_trunc + l,
          ws.tau_scaled, ws.tau_scaled,
          ws.omega_scaled, ws.omega_scaled);
    }

    // Compute B_bar and B_d
    batchedComputeBBarBdKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
        nwav, nlay, l, ws.B_levels, ws.tau_scaled, ws.B_bar, ws.B_d);

    // Doubling
    batchedDoubling(ws, stream, l,
                    solar_flux, solar_mu,
                    ws.tau_above,
                    has_solar, nn_max_global);

    // Adding
    if (!comp_initialized) {
      // First layer: just copy current -> composite
      size_t mat_total = (size_t)nwav * N * N;
      size_t vec_total = (size_t)nwav * N;
      batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
          mat_total, ws.cur_R_ab, ws.comp_R_ab);
      batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
          mat_total, ws.cur_R_ba, ws.comp_R_ba);
      batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
          mat_total, ws.cur_T_ab, ws.comp_T_ab);
      batchedMatCopyKernel<<<divUp(mat_total, BLOCK), BLOCK, 0, stream>>>(
          mat_total, ws.cur_T_ba, ws.comp_T_ba);
      batchedVecCopyKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
          vec_total, ws.cur_s_up, ws.comp_s_up);
      batchedVecCopyKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
          vec_total, ws.cur_s_down, ws.comp_s_down);
      batchedVecCopyKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
          vec_total, ws.cur_s_up_solar, ws.comp_s_up_solar);
      batchedVecCopyKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
          vec_total, ws.cur_s_down_solar, ws.comp_s_down_solar);
      comp_initialized = true;
    } else {
      batchedAdding(ws, stream);
    }
  }

  // --- Boundary intensities ---
  size_t vec_total = (size_t)nwav * N;
  batchedBoundaryIntensityKernel<<<divUp(vec_total, BLOCK), BLOCK, 0, stream>>>(
      nwav, N, ws.B_levels, nlev,
      static_cast<float>(config.top_emission),
      static_cast<float>(config.surface_emission),
      config.use_thermal_emission,
      config.use_diffusion_lower_bc,
      has_surface,
      data.top_emission,
      data.surface_emission,
      data.delta_tau, nlay,
      ws.d_mu_rt,
      ws.I_top_down, ws.I_bot_up);

  // --- Flux reduction ---
  batchedFluxReductionKernel<<<divUp(nwav, BLOCK), BLOCK, 0, stream>>>(
      nwav, N,
      ws.comp_R_ab, ws.comp_T_ba,
      ws.comp_s_up, ws.comp_s_up_solar,
      ws.I_top_down, ws.I_bot_up,
      ws.d_mu_rt, ws.d_wt_rt,
      data.flux_up, data.flux_down, data.flux_direct,
      has_solar, solar_flux, solar_mu, ws.tau_total);

  // Synchronize and cleanup
  cudaStreamSynchronize(stream);
  cublasDestroy(ws.handle);
  freeWorkspace(ws);
}


} // namespace cuda
} // namespace adrt
