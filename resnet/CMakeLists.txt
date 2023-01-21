cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet)

find_package(Torch REQUIRED)

add_executable(load_resnet src/load_resnet.cpp)
target_link_libraries(load_resnet "${TORCH_LIBRARIES}")
set_property(TARGET load_resnet PROPERTY CXX_STANDARD 17)

