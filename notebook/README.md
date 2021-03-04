# Dive into DL with C/C++

["Dive into Deep Learning" ](https://d2l.ai/) is a kind of very fantastic tutorial for beginners who want to get into deep learning tech. It is updated since 2019, and now the latest version is v0.16.1(english edition), and it uses MXNet framework.  

Now most of the developers are using PyTorch or Tensorflow, and they are all supported by python. For the 'traditional' software developers(such as those from communication industry, embedded software industry), C/C++ are more popular. For some special use cases, such as edge computing or embedded computer vision or high performance computing, maybe C/C++ is more useful!

Inspired by ['Dive-into-DL-PyTorch'](https://github.com/ShusenTang/Dive-into-DL-PyTorch) (a github project, re-write all examples by PyTorch), I want to migrate the examples from python to C/C++.

Thanks for developers of 'libtorch'(the c/c++ backend of pytorch) and 'xeus-cling'(a jupyter notebook plugin), I can use c/c++ to write deep learning models and demo the codes in web browser(yes, demo the c/c++ code in jupyter notebook)!

Let's enjoy it!



## compile and install libtorch

Refer the [PyTorch's official guide](https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst):

```sh
git clone -b master --recurse-submodule https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
mkdir pytorch-build
cd pytorch-build
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch
cmake --build . --target install
```





## install and configure the c/c++ plugin for jupyter notebook

```sh
conda create -n your_cpp_env python=3
conda activate your_cpp_env
conda install -c conda-forge xeus-cling



Bingo! You can try c/c++ coding in your notebook!!
```



For include path and shared libs, you can use "**#pragma cling add_library_path**", "**#pragma cling add_include_path**", " **#pragma cling load**",  for further info, please ref the example codes.  

