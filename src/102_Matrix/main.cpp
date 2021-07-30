/*
 * This is the simplest example code from https://eigen.tuxfamily.org/dox/GettingStarted.html
 * */

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {

  // Different ways to initialise matrix
  Matrix<double , 4, 4> m0 = Matrix4d::Ones();

  MatrixXd m1(4, 4);
  m1 = Matrix4d::Zero();

  Matrix4d m2 = Matrix4d::Random(); // [-1, 1]

  Vector4d m3(1.0, 2.0, 3.0, 4.0);
  MatrixXd v = Vector4d::Ones(); // vectors just one dimensional matrices

  VectorXd m4 = Vector4d::Constant(108.5);

  // print
  MatrixXd m [] = {m0, m1, m2, m3, m4};
  const int count = sizeof(m) / sizeof(*m);
  for (int i = 0; i < count; ++i) {
    cout << "m" << i << ": " << endl;
    cout << m[i] << endl << endl;
  }

  //  store orders
  Matrix<double, 4, 4, RowMajor> m5; // default is RowMajor
  m5 << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16;
  cout << "m5" << ": " << endl << m5 << endl;
  cout << "In memory (column-major):" << endl;
  for (int i = 0; i < m5.size(); i++)
    cout << *(m5.data() + i) << "  ";
  cout << endl << endl;

  Matrix<double, 4, 4, ColMajor> m6;
  m6 << 1, 2, 3, 4,  5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16;
  cout << "m6" << ": " << endl << m6 << endl;
  cout << "In memory (column-major):" << endl;
  for (int i = 0; i < m6.size(); i++)
    cout << *(m6.data() + i) << "  ";
  cout << endl << endl;

  // Coefficient accessors
  m0(2, 2) = 100;
  cout << "m0" << ": " << endl << m0 << endl << endl;

  m1 = Matrix4d::Identity(); // overwrite matrix values
  cout << "m1" << ": " << endl << m1 << endl << endl;

  // Matrix Operations
  cout << "m0 + m1: " << endl << m0 + m1 << endl << endl;
  cout << "m0 * m1: " << endl << m0 * m1 << endl << endl;
  cout << "m0 - m1 * 2.2: " << endl << m0 - m1 * 2.2 << endl << endl;
  // Check if two matrices are the same
  cout << "(m0 * m1) == (m0 - m1 * 2.2): " << endl << (m0 * m1 == m0 - m1 * 2.2) << endl << endl;
  cout << "m1 == Matrix4d::Identity(): " << endl << (m1 == Matrix4d::Identity()) << endl << endl;

  cout << "transpose: " << endl << m6.transpose() << endl << endl;
  cout << "inverse: " << endl << m6.inverse() << endl << endl; // if not invertible then shows NaN

  // element-wise
  cout << "element-wise squared: " << endl << m6.array().square() << endl << endl;
  cout << "element-wise multiplication m6 * m6.T: " << endl << m6.array() * m6.transpose().array() << endl << endl;
  cout << "element-wise equal (m0 * m1) == (m0 - m1 * 2.2): " << endl
       << ((m0 * m1).array() == (m0 - m1 * 2.2).array()) << endl << endl;

  // Vector operations
  Vector3f v1, v2;
  v1 = Vector3f::Random();
  v2 = Vector3f::Random();

  cout << "v1: " << endl << v1 << endl << endl;
  cout << "v2: " << endl << v2 << endl << endl;
  cout << "v1 * v2.T: " << endl << v1 * v2.transpose() << endl << endl;
  cout << "v1.T * v2: " << endl << v1.transpose() * v2 << endl << endl;
  cout << "v1.dot(v2): " << endl << v1.dot(v2) << endl << endl;
  cout << "Vector3f(3., 4., 6.).normalized(): " << endl << Vector3f(3., 4., 6.).normalized() << endl << endl;
  cout << "v1.cross(v2): " << endl << v1.cross(v2) << endl << endl;

  Vector4f v3 = v1.homogeneous();
  cout << "v3: " << endl << v3 << endl << endl;
  cout << "v3.hnormalized(): " << endl << v3.hnormalized() << endl << endl;
  // element-wise similar to matrix
  cout << "v1.array().sin(): " << endl << v1.array().sin() << endl << endl;
}
