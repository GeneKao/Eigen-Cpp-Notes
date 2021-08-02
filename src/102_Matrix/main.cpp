/*
 * Refs:
 * https://dritchie.github.io/csci2240/assignments/eigen_tutorial.pdf
 * https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
 * https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
 * */

#include <iostream>
#include <Eigen/Dense>

#define PRINT(x) std::cout << #x << ": " << std::endl << (x) << std::endl << std::endl

int main() {

  using namespace Eigen;

  // Different ways to initialise matrix
  Matrix<double , 4, 4> m0 = Matrix4d::Ones();

  MatrixXd m1(4, 4);
  m1 = Matrix4d::Zero();
  PRINT(m1);
  Matrix4d m2 = Matrix4d::Random(); // [-1, 1]
  PRINT(m2);
  Vector4d m3(1.0, 2.0, 3.0, 4.0);
  PRINT(m3);
  MatrixXd v = Vector4d::Ones(); // vectors just one dimensional matrices
  PRINT(v);
  VectorXd m4 = Vector4d::Constant(108.5);
  PRINT(m4);

  //  store orders
  Matrix<double, 4, 4, RowMajor> m5;
  m5 << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16;
  PRINT(m5);
  std::cout << "In memory (row-major):" << std::endl;
  for (int i = 0; i < m5.size(); i++)
    std::cout << *(m5.data() + i) << "  ";
  std::cout << std::endl << std::endl;

  Matrix<double, 4, 4, ColMajor> m6;
  m6 << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16;
  PRINT(m6);
  std::cout << "In memory (column-major):" << std::endl;
  for (int i = 0; i < m6.size(); i++)
    std::cout << *(m6.data() + i) << "  ";
  std::cout << std::endl << std::endl;

  Matrix<double, 4, 4> m7; // default is Column major
  m7 << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16;
  PRINT(m7);
  std::cout << "In memory (column-major) is default:" << std::endl;
  for (int i = 0; i < m7.size(); i++)
    std::cout << *(m7.data() + i) << "  ";
  std::cout << std::endl << std::endl;

  // Coefficient accessors
  m0(2, 2) = 100;
  PRINT(m0);

  m1 = Matrix4d::Identity(); // overwrite matrix values
  PRINT(m1);

  // Matrix Operations
  PRINT(m0 + m1);
  PRINT(m0 * m1 );
  PRINT( m0 - m1 * 2.2);
  // Check if two matrices are the same
  PRINT(m0 * m1 == m0 - m1 * 2.2);
  PRINT(m1 == Matrix4d::Identity());

  PRINT(m6.transpose());
  PRINT(m6.inverse()); // if not invertible then shows NaN

  // element-wise
  PRINT(m6.array().square());
  PRINT(m6.array() * m6.transpose().array());
  PRINT(((m0 * m1).array() == (m0 - m1 * 2.2).array()));

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
}
