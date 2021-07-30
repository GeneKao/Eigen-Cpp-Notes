/*
 * First code to test if Eigen is installed correctly.
 *
 */

#include <iostream>
#include <Eigen/Core>

using namespace Eigen;

int main()
{
  std::cout << "Eigen version: " << EIGEN_MAJOR_VERSION << "."
            << EIGEN_MINOR_VERSION << std::endl;
  return 0;
}
