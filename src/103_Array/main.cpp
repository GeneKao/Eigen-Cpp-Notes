/*
 * Array can  perform coefficient-wise operations, which might not have a linear algebraic meaning.
 * Refs:
 * https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html
 * */

#include <iostream>
#include <Eigen/Dense>

#define PRINT(x) std::cout << #x << ": " << std::endl << (x) << std::endl << std::endl
#define PRINT_SIZE(x) std::cout << #x << " is of size " << x.rows() << "x" << x.cols() << std::endl << std::endl
#define SECTION(x) std::cout << "======================== " << x << " =======================" << std::endl << std::endl

int main() {

  using namespace Eigen;

  SECTION("Initialisations");
  ArrayXXf m1(2,2); // Array<double,Dynamic,Dynamic>
  m1 = Array22f::Ones();
  PRINT(m1);
  Array22f m2;
  m2 << 1, 2, 3, 4;

  // https://eigen.tuxfamily.org/dox/group__TutorialAdvancedInitialization.html
  ArrayXXf table(10, 4);
  table.col(0) = ArrayXf::LinSpaced(10, 0, 90);
  table.col(1) = M_PI / 180 * table.col(0);
  table.col(2) = table.col(1).sin();
  table.col(3) = table.col(1).cos();
  std::cout << "table: " << std::endl;
  std::cout << "\tDegrees\t\tRadians\t\tSine\t\tCosine\n";
  std::cout << table << std::endl;

  SECTION("Operations");
  PRINT(m1+5);
  PRINT(m1+m2);
  PRINT(m1*2);
  PRINT(m1*m2);

  Array33f m3 = Array33f::Random();
  PRINT(m3);
  PRINT(m3.abs().sqrt());
  PRINT(m3.min(m3.abs().sqrt()));
  // more coefficient-wise & Array operators https://eigen.tuxfamily.org/dox/group__QuickRefPage.html

  SECTION("Types converting");
  // convert type
  MatrixXf m(2,2);
  MatrixXf n(2,2);
  MatrixXf result(2,2);
  m << 1,2,
  3,4;
  n << 5,6,
  7,8;
  PRINT(m);
  PRINT(n);
  result = m * n;
  std::cout << "-- Matrix m*n: --" << std::endl << result << std::endl << std::endl;
  result = m.array() * n.array();
  std::cout << "-- Array m*n: --" << std::endl << result << std::endl << std::endl;
  result = m.cwiseProduct(n);
  std::cout << "-- With cwiseProduct: --" << std::endl << result << std::endl << std::endl;
  result = m.array() + 4;
  std::cout << "-- Array m + 4: --" << std::endl << result << std::endl << std::endl;
  result = (m.array() + 4).matrix() * m;
  std::cout << "-- Combination 1: --" << std::endl << result << std::endl << std::endl;
  result = (m.array() * n.array()).matrix() * m;
  std::cout << "-- Combination 2: --" << std::endl << result << std::endl << std::endl;

  ArrayXXd arr_result = result.array().cast<double>();
  PRINT(arr_result);
}
