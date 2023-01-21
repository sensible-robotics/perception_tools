cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(lenet)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(load_lenet cpp/load_lenet.cpp)
target_link_libraries(load_lenet "${TORCH_LIBRARIES}")
set_property(TARGET load_lenet PROPERTY CXX_STANDARD 17)


add_executable(test_lenet cpp/test_lenet.cpp)
target_link_libraries(test_lenet "${TORCH_LIBRARIES}" ${OpenCV_LIBRARIES})
set_property(TARGET test_lenet PROPERTY CXX_STANDARD 17)

