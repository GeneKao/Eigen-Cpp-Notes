/*
 * Refs:
 * https://eigen.tuxfamily.org/dox/index.html
 * https://dritchie.github.io/csci2240/assignments/eigen_tutorial.pdf
 * */

#include <iostream>
#include <Eigen/Dense>

#define PRINT(x) std::cout << #x << ": " << std::endl << (x) << std::endl << std::endl
#define PRINT_SIZE(x) std::cout << #x << " is of size " << x.rows() << "x" << x.cols() << std::endl << std::endl
#define SECTION(x) std::cout << "======================== " << x << " =======================" << std::endl << std::endl

int main() {

  using namespace Eigen;

  SECTION("Initialisations");
  // Different ways to initialise matrix
  Matrix<double , Dynamic, Dynamic, 0, 6, 8> m0 = Matrix4d::Ones();
  // the last three are optional, so we don't have to worry about it most of the time.
  // isRowMajor=0: default is column major, MaxRowsAtCompileTime, MaxColsAtCompileTime
  PRINT(m0);
  MatrixXd m1(4, 4); // typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
  m1 = Matrix4d::Zero();
  PRINT(m1);
  // fix size matrix
  Matrix4d m2 = Matrix4d::Random(); // [-1, 1]
  PRINT(m2);
  Vector4d m3(1.0, 2.0, 3.0, 4.0); // typedef Matrix<double, 4, 1> Vector3d;
  PRINT(m3);
  MatrixXd v = Vector4d::Ones(); // vectors just one dimensional matrices
  PRINT(v);
  VectorXd m4 = Vector4d::Constant(108.5); // column vector
  PRINT(m4);
  RowVectorXd m4r = Vector4d::Constant(108.5); // row vector
  PRINT(m4r);

  SECTION("Store orders");
  // store orders
  // https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
  Matrix<double, 4, 4, RowMajor> m5;
  m5 << 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16; // comma-initializer syntax
  PRINT(m5);
  std::cout << "In memory (row-major):" << std::endl;
  for (int i = 0; i < m5.size(); i++)
    std::cout << *(m5.data() + i) << "  ";
  std::cout << std::endl << std::endl;

  Matrix<double, 4, 4, ColMajor> m6;
  m6 << 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  PRINT(m6);
  std::cout << "In memory (column-major):" << std::endl;
  for (int i = 0; i < m6.size(); i++)
    std::cout << *(m6.data() + i) << "  ";
  std::cout << std::endl << std::endl;

  Matrix<double, 4, 4> m7; // default is Column major
  m7 << 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  PRINT(m7);
  std::cout << "In memory (column-major) is default:" << std::endl;
  for (int i = 0; i < m7.size(); i++)
    std::cout << *(m7.data() + i) << "  ";
  std::cout << std::endl << std::endl;

  // ref: https://eigen.tuxfamily.org/dox/group__TutorialAdvancedInitialization.html
  MatrixXd m8(4, 8);
  m8 << m6, m7;
  PRINT(m8);

  MatrixXd m9(8, 4);
  m9 << m6, m7;
  PRINT(m9);

  SECTION("Block Operation");
  // block operation https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
  MatrixXd m10(5, 9);
  m10.row(0) << RowVectorXd::Ones(9)*100;
  m10.block(1, 0, 4, 4) << m6;
  m10.block<4, 4>(1, 4) = m7*2;
  m10.col(8).tail(4) << VectorXd::Ones(4)*99; // col and row are special case of block
  PRINT(m10);
  // some other keywords:
  // topLeftCorner, bottomLeftCorner, topRightCorner, bottomRightCorner, topRows, bottomRows, leftCols, rightCols.
  PRINT(m5);
  PRINT(m5.leftCols(2));
  PRINT(m5.bottomRows<2>());
  m5.topLeftCorner(1,3) = m5.bottomRightCorner(3,1).transpose();
  PRINT(m5);
  VectorXd v0(6);
  v0 << 1, 2, 3, 4, 5, 6;
  PRINT(v0.head(3));
  PRINT(v0.tail<3>());
  v0.segment(1,4) *= 2;
  PRINT(v0);

  SECTION("Diagonal");
  // Coefficient accessors
  m0(2, 2) = 100;
  PRINT(m0);
  PRINT(m0.diagonal());
  MatrixXd m0d1 = m0.diagonal().asDiagonal();
  // https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html#a14235b62c90f93fe910070b4743782d0
  PRINT(m0d1);
  DiagonalMatrix<double, Dynamic> m0d2;
  // http://eigen.tuxfamily.org/dox/classEigen_1_1DiagonalMatrix.html
  m0d2.diagonal() = m0.diagonal();
  PRINT(MatrixXd(m0d2));

  m1 = Matrix4d::Identity(); // overwrite matrix values
  PRINT(m1);

  SECTION("Vector operations");
  // Vector operations
  Vector3f v1, v2;
  v1 = Vector3f::Random();
  v2 = Vector3f::Random();
  PRINT(v1);
  PRINT(v2);
  PRINT(v1 * v2.transpose());
  PRINT(v1.transpose() * v2);
  PRINT(v1.dot(v2));
  PRINT(v1.cross(Vector3f(3., 4., 6.).normalized()));
  PRINT(v1.cross(v2));

  Vector4f v3 = v1.homogeneous();
  PRINT(v3);
  PRINT(v3.hnormalized());
  // element-wise similar to matrix
  PRINT(v1.array().sin());

  SECTION("Matrix Operations");
  // Matrix Operations
  PRINT(m0 + m1);
  PRINT(m0 * m1);
  PRINT(m1 += m1);
  PRINT(m1 *= 2);
  PRINT(m0 - m1 * 2.2);
  // Check if two matrices are the same
  PRINT(m0 * m1 == m0 - m1 * 2.2);
  PRINT(m1 == Matrix4d::Identity()*4);

  PRINT(m6.transpose()); // this doesn't modify m6
  // NEVER DO "a = a.transpose()" aliasing issue: https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html
  // but matrix operation is fine, no problem! a = a*a;
  // https://eigen.tuxfamily.org/dox/group__TopicAliasing.html
  m6.transposeInPlace(); // we can do this instead
  PRINT(m6);
  PRINT(m6.inverse()); // if not invertible then shows NaN

  SECTION("Element-Wise");
  // element-wise
  PRINT(m6.array().square());
  PRINT(m6.array() * m6.transpose().array());
  PRINT(((m0 * m1).array() == (m0 - m1 * 2.2).array()));

  SECTION("Basic arithmetic reduction operations");
  // Basic arithmetic reduction operations
  // https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html
  // https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
  PRINT(m1);
  PRINT(m1.sum());
  PRINT(m1.prod());
  PRINT(m1.mean());
  PRINT(m1.minCoeff());
  PRINT(m1.maxCoeff());
  PRINT(m1.trace());
  // visitors
  MatrixXd::Index maxRow, maxCol;
  double maxOfm1 = m1.maxCoeff(&maxRow, &maxCol);
  std::cout << maxOfm1 << " is at position: (" << maxRow << "," << maxCol << ")" << std::endl << std::endl;
  std::ptrdiff_t i, j;
  double minOfm1 = m1.minCoeff(&i, &j);
  std::cout << minOfm1 << " is at position: (" << i << "," << j << ")" << std::endl << std::endl;
  // partial reductions
  PRINT(m1.colwise().maxCoeff());
  PRINT(m1.rowwise().maxCoeff());
  PRINT(m1.cwiseSqrt());
  PRINT(m1.cwiseSqrt().colwise().maxCoeff());

  SECTION("Norm");
  PRINT(m1.squaredNorm());
  PRINT(m1.norm());
  // lp-norm
  PRINT(m1.lpNorm<2>());
  PRINT(m1.lpNorm<1>());
  PRINT(m1.lpNorm<Infinity>());
  // Operator norm: https://en.wikipedia.org/wiki/Operator_norm
  // 1-norm(m1)
  PRINT(m1.cwiseAbs().colwise().sum().maxCoeff());
  PRINT(m1.colwise().lpNorm<1>().maxCoeff());
  // Infinity-norm(m1)
  PRINT(m1.cwiseAbs().rowwise().sum().maxCoeff());
  PRINT(m1.rowwise().lpNorm<1>().maxCoeff());

  PRINT(m6);
  MatrixXd::Index maxIndex;
  double maxNorm = m6.colwise().sum().maxCoeff(&maxIndex);
  std::cout << "Maximum sum at position " << maxIndex << std::endl
            << "The corresponding vector is: " << std::endl
            << m6.col(maxIndex) << std::endl
            << "And its sum is is: " << maxNorm << std::endl << std::endl;

  SECTION("Broadcasting");
  // https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
  MatrixXd mb = MatrixXd::Zero(2, 4);
  VectorXd vb = Vector2d(0, 1);
  PRINT(mb);
  PRINT(vb);
  mb.colwise() += vb;
  PRINT(mb);
  mb.setZero();
  vb = Vector4d(0, 1, 2, 3);
  mb.rowwise() += vb.transpose();
  PRINT(mb);
  PRINT(vb);
  // find nearest neighbour
  mb.setRandom();
  vb.setRandom(2);
  PRINT(mb);
  PRINT(vb);
  MatrixXf::Index index;
  (mb.colwise() - vb).colwise().squaredNorm().minCoeff(&index);
  std::cout << "Nearest neighbour is column " << index << ":" << std::endl;
  std::cout << mb.col(index) << std::endl;

  SECTION("Resizing");
  // resizing
  // https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
  VectorXd v4 = Vector4d::Ones()*100;
  PRINT(v4);
  v4.conservativeResize(6); // this leave the old value untouched
  PRINT(v4);
  v4.resize(8);
  PRINT(v4);

  PRINT_SIZE(m1);
  m1 = MatrixXd::Identity(6, 6); // copy and resize of matrix with dynamic size
  PRINT(m1);
  PRINT_SIZE(m1);

  // change type of matrix
  MatrixXf m11 = m1.cast<float>();
  PRINT(m11);

  SECTION("Temporary objects");
  // using comma-initializer as temporary objects
  // https://eigen.tuxfamily.org/dox/group__TutorialAdvancedInitialization.html
  MatrixXf mat = MatrixXf::Random(2, 3);
  PRINT(mat);
  mat = (MatrixXf(2,2) << 0, 1, 1, 0).finished() * mat;
  PRINT(mat);
  // https://eigen.tuxfamily.org/dox/structEigen_1_1CommaInitializer.html#a3cf9e2b8a227940f50103130b2d2859a

}
