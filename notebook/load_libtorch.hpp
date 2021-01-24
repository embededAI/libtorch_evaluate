/*
 * =====================================================================================
 *
 *       Filename:  load_libtorch.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/24/2021 03:01:43
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef LOAD_LIBTORCH_HPP
#define LOAD_LIBTORCH_HPP

#pragma cling add_library_path("/usr/local/lib")
#pragma cling add_include_path("/usr/local/include")

#if defined(__linux__)

#pragma cling add_library_path("/home/ubuntu/liuhang/libtorch/pytorch-install/lib")
#pragma cling add_include_path("/home/ubuntu/liuhang/libtorch/pytorch-install/include")
#pragma cling add_include_path("/home/ubuntu/liuhang/libtorch/pytorch-install/include/torch/csrc/api/include")

#pragma cling load("libc10_cuda.so")
#pragma cling load("libc10.so")
#pragma cling load("libtorch.so")

#elif __APPLE__

#pragma cling add_library_path("/Users/liuhang/codes/github/libtorch/release/lib")
#pragma cling add_include_path("/Users/liuhang/codes/github/libtorch/release/include")
#pragma cling add_include_path("/Users/liuhang/codes/github/libtorch/release/include/torch/csrc/api/include")

#pragma cling load("libtorch.dylib")

#endif


#endif /*LOAD_LIBTORCH_HPP*/
