/// @file matrix.h
/// @brief Dense NxN matrix classes for the adding-doubling RT solver.
///
/// Provides both a fixed-size templated Matrix<N> (using Eigen fixed-size types
/// for stack allocation, SIMD, and compile-time unrolling) and a DynamicMatrix
/// (using Eigen::MatrixXd) for runtime-sized operations.

#pragma once

#include <Eigen/Dense>

#include <cassert>
#include <vector>


namespace adrt {

// ============================================================================
//  Fixed-size dense square matrix (compile-time N)
// ============================================================================

template<int N>
class Matrix {
public:
  using EigenMat = Eigen::Matrix<double, N, N>;
  using EigenVec = Eigen::Matrix<double, N, 1>;

  // --- Construction --------------------------------------------------------

  /// Zero-initialised NxN matrix.
  Matrix() : mat_(EigenMat::Zero()) {}

  /// Construct from an Eigen matrix.
  explicit Matrix(const EigenMat& m) : mat_(m) {}
  explicit Matrix(EigenMat&& m) : mat_(std::move(m)) {}

  /// Return an NxN identity matrix.
  static Matrix identity() {
    Matrix m;
    m.mat_ = EigenMat::Identity();
    return m;
  }

  /// Return a diagonal matrix from a vector of values.
  static Matrix diagonal(const EigenVec& v) {
    Matrix m;
    m.mat_ = v.asDiagonal();
    return m;
  }

  // --- Element access ------------------------------------------------------

  static constexpr int size() { return N; }

  double& operator()(int i, int j) { return mat_(i, j); }
  double  operator()(int i, int j) const { return mat_(i, j); }

  // --- Direct Eigen access -------------------------------------------------

  EigenMat&       eigen()       { return mat_; }
  const EigenMat& eigen() const { return mat_; }

  // --- Arithmetic (return new matrix) --------------------------------------

  /// C = A * B
  Matrix multiply(const Matrix& B) const {
    Matrix C;
    C.mat_.noalias() = mat_ * B.mat_;
    return C;
  }

  /// y = A * x  (matrix-vector product, fixed-size)
  EigenVec multiply(const EigenVec& x) const {
    return mat_ * x;
  }

  /// C = A + alpha * B
  Matrix add(const Matrix& B, double alpha = 1.0) const {
    return Matrix(EigenMat(mat_ + alpha * B.mat_));
  }

  /// C = alpha * A
  Matrix scale(double alpha) const {
    return Matrix(EigenMat(alpha * mat_));
  }

  // --- In-place operations -------------------------------------------------

  void addInplace(const Matrix& B, double alpha = 1.0) {
    mat_ += alpha * B.mat_;
  }

  void scaleInplace(double alpha) {
    mat_ *= alpha;
  }

  void setZero() {
    mat_.setZero();
  }

  // --- Linear algebra ------------------------------------------------------

  /// Solve A x = b for x.
  EigenVec solve(const EigenVec& b) const {
    return mat_.partialPivLu().solve(b);
  }

  /// Compute the inverse.
  Matrix inverse() const {
    return Matrix(EigenMat(mat_.inverse()));
  }

  /// Solve A X = B (multiple RHS), returns X = A^{-1} B.
  Matrix solveMatrix(const Matrix& B) const {
    return Matrix(EigenMat(mat_.partialPivLu().solve(B.mat_)));
  }

  /// Solve X A = B, returns X = B A^{-1}.
  /// Equivalent to solving A^T X^T = B^T.
  Matrix rightSolveMatrix(const Matrix& B) const {
    Matrix result;
    result.mat_ = mat_.transpose().partialPivLu().solve(B.mat_.transpose()).transpose();
    return result;
  }

private:
  EigenMat mat_;
};


// ============================================================================
//  Dynamic-size dense square matrix (runtime N)
// ============================================================================

class DynamicMatrix {
public:
  explicit DynamicMatrix(int n) : mat_(Eigen::MatrixXd::Zero(n, n)) {}
  explicit DynamicMatrix(const Eigen::MatrixXd& m) : mat_(m) {}
  explicit DynamicMatrix(Eigen::MatrixXd&& m) : mat_(std::move(m)) {}

  int size() const { return static_cast<int>(mat_.rows()); }

  double& operator()(int i, int j) { return mat_(i, j); }
  double  operator()(int i, int j) const { return mat_(i, j); }

  Eigen::MatrixXd&       eigen()       { return mat_; }
  const Eigen::MatrixXd& eigen() const { return mat_; }

  static DynamicMatrix identity(int n) {
    return DynamicMatrix(Eigen::MatrixXd::Identity(n, n));
  }

  static DynamicMatrix diagonal(const std::vector<double>& v) {
    int n = static_cast<int>(v.size());
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
      m(i, i) = v[i];
    return DynamicMatrix(std::move(m));
  }

  DynamicMatrix multiply(const DynamicMatrix& B) const {
    Eigen::MatrixXd C(mat_.rows(), mat_.cols());
    C.noalias() = mat_ * B.mat_;
    return DynamicMatrix(std::move(C));
  }

  std::vector<double> multiply(const std::vector<double>& x) const {
    int n = size();
    Eigen::Map<const Eigen::VectorXd> xv(x.data(), n);
    Eigen::VectorXd y = mat_ * xv;
    return std::vector<double>(y.data(), y.data() + n);
  }

  DynamicMatrix add(const DynamicMatrix& B, double alpha = 1.0) const {
    return DynamicMatrix(Eigen::MatrixXd(mat_ + alpha * B.mat_));
  }

  DynamicMatrix scale(double alpha) const {
    return DynamicMatrix(Eigen::MatrixXd(alpha * mat_));
  }

  void addInplace(const DynamicMatrix& B, double alpha = 1.0) {
    mat_ += alpha * B.mat_;
  }

  void scaleInplace(double alpha) { mat_ *= alpha; }
  void setZero() { mat_.setZero(); }

  std::vector<double> solve(const std::vector<double>& b) const {
    int n = size();
    Eigen::Map<const Eigen::VectorXd> bv(b.data(), n);
    Eigen::VectorXd x = mat_.partialPivLu().solve(bv);
    return std::vector<double>(x.data(), x.data() + n);
  }

  DynamicMatrix inverse() const {
    return DynamicMatrix(Eigen::MatrixXd(mat_.inverse()));
  }

  DynamicMatrix solveMatrix(const DynamicMatrix& B) const {
    return DynamicMatrix(Eigen::MatrixXd(mat_.partialPivLu().solve(B.mat_)));
  }

  /// Solve X A = B, returns X = B A^{-1}.
  DynamicMatrix rightSolveMatrix(const DynamicMatrix& B) const {
    Eigen::MatrixXd result = mat_.transpose().partialPivLu().solve(B.mat_.transpose()).transpose();
    return DynamicMatrix(std::move(result));
  }

private:
  Eigen::MatrixXd mat_;
};

} // namespace adrt
