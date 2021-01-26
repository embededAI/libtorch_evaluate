/*
 * =====================================================================================
 *
 *       Filename:  torch_tensor.cpp
 *
 *    Description:  evaluate tensor ops in libtorch
 *
 *        Version:  1.0
 *        Created:  01/18/2021 23:57:50
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liu Hang (liuhang20011@163.com)
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <torch/torch.h>
#include <vector>
#include <iostream>



int main(int argc, char* argv[])
{
  torch::Tensor x = torch::zeros({3,4});

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      x[i][j] = i*4+j;
    }
  }
      
  std::cout << "x =" << std::endl;
  std::cout << (x) << std::endl << std::endl;

  std::vector<int32_t> v = {1,2,7,8}; 
  auto idx = torch::from_blob(v.data(), v.size(), torch::kInt32);
  std::cout << (idx) << std::endl << std::endl;

  std::cout << "<<< tensor.gather >>>" << std::endl;
  torch::Tensor g = torch::ones({2,2}, torch::kInt64);
  std::cout << (x.gather(-1, g)) << std::endl << std::endl;
  // std::cout << (torch::gather(x, 0, g, false)) << std::endl << std::endl;
  //

  return 0;
}
