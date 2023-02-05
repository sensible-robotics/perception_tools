cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(load_resnet cpp/load_resnet.cpp)
target_link_libraries(load_resnet "${TORCH_LIBRARIES}")
set_property(TARGET load_resnet PROPERTY CXX_STANDARD 17)

add_executable(test_resnet cpp/test_resnet.cpp)
target_link_libraries(test_resnet "${TORCH_LIBRARIES}" ${OpenCV_LIBRARIES})
set_property(TARGET test_resnet PROPERTY CXX_STANDARD 17)
