# dcgan - a simple example with libtorch

This is an official example offered by pytorch.org, which use libtorch to show how to build a simple NN model with C++ by yourself.



## 1.build libtorch

Please refer the build [instructions](https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst).

I checkout the v1.7.1 source code of pytorch by 

```git clone -b v1.7.1 --recurse-submodule https://github.com/pytorch/pytorch.git```

```cd pytorch ```

```git submodule update --init --recursive```

Then I build the source code with cmake(the 2nd way in web page above).



## 2.write CMakeList file

I refer the official template, but add something new into it: define 'Torch_DIR' in cmakelist file.



## 3.compile the code

```mkdir build ```

```cd build```

```cmake ..```

```make```

















[by the way, using 'Typora' to edit markdown file is really good!]
