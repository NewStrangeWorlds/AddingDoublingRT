/// @file layer.hpp
/// @brief Layer data structures for the adding-doubling RT solver.
///
/// Holds the reflection/transmission matrices and source vectors for a single
/// layer or a composite of layers, in both fixed-size and dynamic-size forms.

#pragma once

#include "matrix.hpp"

#include <vector>

namespace adrt {

// ============================================================================
//  Fixed-size LayerMatrices (compile-time N)
// ============================================================================

template<int N>
struct LayerMatrices {
  using Vec = typename Matrix<N>::EigenVec;

  Matrix<N> R_ab;
  Matrix<N> R_ba;
  Matrix<N> T_ab;
  Matrix<N> T_ba;

  Vec s_up;
  Vec s_down;
  Vec s_up_solar;
  Vec s_down_solar;

  /// Thermal source-derivative basis for analytic temperature Jacobians.
  /// For a layer bounded by level Planck values B_top, B_bottom:
  ///   d(s_up)/d(B_top)    = j_p,   d(s_up)/d(B_bottom)    = j_q,
  ///   d(s_down)/d(B_top)  = j_q,   d(s_down)/d(B_bottom)  = j_p.
  /// Both are temperature-independent (built from the doubling vectors y, z or
  /// the pure-absorption coefficients). Zero unless populated by doubling<N>.
  Vec j_p;
  Vec j_q;

  bool is_scattering = false;

  LayerMatrices()
    : T_ab(Matrix<N>::identity()), T_ba(Matrix<N>::identity()),
      s_up(Vec::Zero()), s_down(Vec::Zero()),
      s_up_solar(Vec::Zero()), s_down_solar(Vec::Zero()),
      j_p(Vec::Zero()), j_q(Vec::Zero()) {}
};


// ============================================================================
//  Dynamic-size LayerMatrices (runtime N)
// ============================================================================

struct DynLayerMatrices {
  DynamicMatrix R_ab;
  DynamicMatrix R_ba;
  DynamicMatrix T_ab;
  DynamicMatrix T_ba;

  std::vector<double> s_up;
  std::vector<double> s_down;
  std::vector<double> s_up_solar;
  std::vector<double> s_down_solar;

  /// Thermal source-derivative basis (see LayerMatrices<N>::j_p/j_q).
  std::vector<double> j_p;
  std::vector<double> j_q;

  bool is_scattering = false;

  explicit DynLayerMatrices(int n)
    : R_ab(n), R_ba(n),
      T_ab(DynamicMatrix::identity(n)), T_ba(DynamicMatrix::identity(n)),
      s_up(n, 0.0), s_down(n, 0.0),
      s_up_solar(n, 0.0), s_down_solar(n, 0.0),
      j_p(n, 0.0), j_q(n, 0.0) {}
};

} // namespace adrt
