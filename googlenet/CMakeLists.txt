cmake_minimum_required(VERSION 2.8)
project(googlenet)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(load_googlenet cpp/load_googlenet.cpp)
target_link_libraries(load_googlenet "${TORCH_LIBRARIES}")
set_property(TARGET load_googlenet PROPERTY CXX_STANDARD 17)

add_executable(test_googlenet cpp/test_googlenet.cpp)
target_link_libraries(test_googlenet "${TORCH_LIBRARIES}" ${OpenCV_LIBRARIES})
set_property(TARGET test_googlenet PROPERTY CXX_STANDARD 17)
