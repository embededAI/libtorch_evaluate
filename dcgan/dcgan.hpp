/*
 * =====================================================================================
 *
 *       Filename:  dcgan.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/15/2021 11:24:02 AM
 *       Revision:  none
 *       Compiler:  gcc/g++
 *
 *         Author:  Liu Hang (lh), liuhang@aiostchina.com
 *   Organization:  A.I.O.S.T
 *
 * =====================================================================================
 */

#ifndef DCGAN_HPP
#define DCGAN_HPP

#include <ATen/Functions.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/options/conv.h>
#include <torch/torch.h>


/* ***
 *  a Net module inherited with torch::nn::Module
 *  without registered module
 * */
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    b = register_parameter("b", torch::randn(M));
  }

  torch::Tensor forward(torch::Tensor input) {
    return torch::addmm(b, input, W);
  }

  torch::Tensor W, b;

};


/* ***
 *  a Net module inherited with torch::nn::Module
 *  without registered module
 * */
struct Net_a : torch::nn::Module {
//  Net_a(int64_t N, int64_t M) : linear(register_module("linear", torch::nn::Linear(N, M)))
  Net_a(int64_t N, int64_t M) 
  {
    linear = register_module("linear", torch::nn::Linear(N, M));
    bias = register_parameter("bias", torch::randn(M));
  }

  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + bias;
  }

  torch::nn::Linear linear{nullptr};
  torch::Tensor bias;

};


/* ***
 *  DCGAN
 * */

struct DCGANGeneratorImpl : torch::nn::Module {
  DCGANGeneratorImpl(int kNoiseSize)
    : conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).stride(1).padding(0).bias(false)),
      batch_norm1(256),
      conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
      batch_norm2(128),
      conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
      batch_norm3(64),
      conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))
  {

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(batch_norm1(conv1(x)));
    x = torch::relu(batch_norm2(conv2(x)));
    x = torch::relu(batch_norm3(conv3(x)));
    x = torch::tanh((conv4(x)));
    return x;
  }

  torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
  torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANGenerator);


struct DCGANDiscriminatorImpl : torch::nn::Module {
  DCGANDiscriminatorImpl()
    : conv1(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
      conv2(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
      batch_norm2(128),
      conv3(torch::nn::Conv2dOptions(128,256, 4).stride(2).padding(1).bias(false)),
      batch_norm3(256),
      conv4(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false))
  {

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::leaky_relu(conv1(x), 0.2);
    x = torch::leaky_relu(batch_norm2(conv2(x)), 0.2);
    x = torch::leaky_relu(batch_norm3(conv3(x)), 0.2);
    x = torch::sigmoid(conv4(x));
    return x;
  }

  torch::nn::Conv2d conv1, conv2, conv3, conv4;
  torch::nn::BatchNorm2d batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANDiscriminator);








#endif /*DCGAN_HPP*/
