/*
 * =====================================================================================
 *
 *       Filename:  dcgan.cpp
 *
 *    Description:  a DL test programme via libtorch 
 *
 *        Version:  1.0
 *        Created:  01/14/2021 04:22:42 PM
 *       Revision:  none
 *       Compiler:  gcc/g++
 *
 *         Author:  Liu Hang (lh), liuhang@aiostchina.com
 *   Organization:  A.I.O.S.T
 *
 * =====================================================================================
 */
#include <ATen/Functions.h>
#include <c10/core/Device.h>
#include <stdlib.h>
#include <torch/torch.h>
#include <iostream>
#include "dcgan.hpp"
#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>



/* *****
 *
 *  global defs 
 *
 * */
#define SW_VER "v0.1"
#define LHDEBUG 1
#define dPrint(...) \
         do {printf("%s - %d: %s\r\n", __func__, __LINE__, __VA_ARGS__);} while(0)

/* *****
 *
 *  global vars 
 *
 * */
int64_t kBatchSize = 1;
int64_t kNumberOfEpochs = 30;
int64_t kNoiseSize = 100;
int64_t kCheckpointEvery = 200;
int64_t kNumberOfSamplesPerCheckpoint = 10;

/* *****
 *
 *  local functions
 *
 * */
static void libtorch_simple_demo(void); 


DEFINE_bool (demo, false, "run a simple demo to show Tensor and Module");
DEFINE_bool (showSample, false, "show samples of datasets");
DEFINE_string (input, "./mnist", "set the mnist dataset paht");
DEFINE_string (device, "CPU", "set the device which to run libtorch(CPU|GPU)");

int main(int argc, char** argv)
{
  gflags::SetUsageMessage(" run dcgan network, train and test \n"
    "usage: dcgan <option>=<args>\n\n"
    "options:\n"
    " demo         a small demo for how to use torch::nn::module\n"
    " input        set the dataset's path(default: ./mnist)\n"
    " device       set the device type, GPU or CPU\n"
    " showSample   show sample details for the input dataset(mnist)\n"
    "==================\n");

  gflags::SetVersionString(SW_VER);

  if (argc < 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "build/dcgan");
    return 0;
  }

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string dataset_path;

  if (FLAGS_demo) {
    std::cout << "run demo ..." << std::endl ;
    libtorch_simple_demo();
     return 0;
  }

  if (!FLAGS_input.empty()) {
    std::cout << "mnist dataset : " + FLAGS_input << std::endl;
    dataset_path = FLAGS_input;
  } else {
    // default path
    dataset_path = "./mnist";
  }

  torch::Device device(torch::kCPU);
  if (FLAGS_device == "GPU") {
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Training on GPU." << std::endl;
      device = torch::Device(torch::kCUDA);
    }
  }

  DCGANGenerator generator(kNoiseSize);
  generator->to(device);
  DCGANDiscriminator discriminator;
  discriminator->to(device);
  
  auto dataset = torch::data::datasets::MNIST(dataset_path)
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

  const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

  auto data_loader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

  if (FLAGS_showSample) {
    for (torch::data::Example<>& batch : *data_loader) {
      std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
      for (int64_t i = 0; i < batch.data.size(0); ++i) {
        std::cout << batch.target[i].item<int64_t>() << " ";
      }
      std::cout << std::endl;
    }
  }

  torch::optim::Adam generator_optimizer(
    generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));
  torch::optim::Adam discriminator_optimizer(
    discriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple (0.5, 0.5)));

  int64_t checkpoint_counter = 1;

  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();
  
      // Train discriminator with fake images.
      torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();
  
      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();
  
      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
  
      std::printf(
          "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
          epoch,
          kNumberOfEpochs,
          ++batch_index,
          batches_per_epoch,
          d_loss.item<float>(),
          g_loss.item<float>());
    }

    if (batch_index % kCheckpointEvery == 0) {
      // Checkpoint the model and optimizer state.
      torch::save(generator, "generator-checkpoint.pt");
      torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
      torch::save(discriminator, "discriminator-checkpoint.pt");
      torch::save(
          discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
      // Sample the generator and save the images.
      torch::Tensor samples = generator->forward(torch::randn(
          {kNumberOfSamplesPerCheckpoint, kNoiseSize, 1, 1}, device));
      torch::save(
          (samples + 1.0) / 2.0,
          torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
      std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
    } 
  }
  
  gflags::ShutDownCommandLineFlags();
  return 0;
}


static void libtorch_simple_demo(void) 
{
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;

  std::cout << "<-------- split line -------->" << std::endl << std::endl;

  Net net(4, 5);
  for (const auto& p : net.parameters()) {
    std::cout << p << std::endl;
  }

  std::cout << "<-------- split line -------->" << std::endl << std::endl;

  for (const auto& odict : net.named_parameters()) {
    std::cout << "Param: " << odict.key() << "\nValue: \n" << odict.value() << std::endl;
    std::cout << std::endl;
  }

  std::cout << "<-------- split line -------->" << std::endl << std::endl;

  Net_a net_a(5, 6);
  for (const auto& p : net_a.named_parameters()) {
    std::cout << "Param: " << p.key() << "\nValue: \n" << p.value() << std::endl;
    std::cout << std::endl;
  }

  return ;
}





