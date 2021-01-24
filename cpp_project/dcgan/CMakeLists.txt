cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)


if(NOT DEFINED ENV{Torch_DIR})
  set( ENV{Torch_DIR} "${PROJECT_SOURCE_DIR}/../../pytorch-install/share/cmake" )
endif()

##
#  compile libtorch with :
#  https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst
#  all result can be found in 'pytorch-install' folder
##
find_package(gflags REQUIRED)
find_package(Torch REQUIRED ${TORCH_PATH})
find_package(OpenCV REQUIRED
  COMPONENTS
  opencv_core
  opencv_imgproc
  opencv_highgui
)

include_directories(
    "."
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan gflags "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)
