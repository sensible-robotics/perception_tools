cmake_minimum_required(VERSION 2.8.3)
project(perception_tools)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  rospy
)


# Python for toppra calls
find_package(PythonLibs REQUIRED)


catkin_package(
  INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} #include
  LIBRARIES #craftsman_search
  CATKIN_DEPENDS rospy 
  DEPENDS 
)

include_directories(
#  include
  ${PYTHON_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)


#add_library(craftsman_search
#  src/cartesian_trajectories.cpp
#  src/joint_a_star_bidirectional.cpp
#)

#target_link_libraries(craftsman_search ${catkin_LIBRARIES} ${OMPL_LIBRARIES} lz4)

# add_executable(test_utils src/test.cpp)
# target_link_libraries(test_utils ${catkin_LIBRARIES} ${TRAC_IK_LIBRARY_NAME} ${PYTHON_LIBS})


#install(TARGETS craftsman_search ${CRAFTSMAN_SEARCH_SCRIPTS}  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

#install(DIRECTORY include/${PROJECT_NAME}/
#  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

#find_package(Python COMPONENTS Interpreter Development)


#if("$ENV{ROS_DISTRO}" STRGREATER "melodic")
#  execute_process(COMMAND /bin/sh -c "python3 -m compileall scripts"
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#else()
#  execute_process(COMMAND /bin/sh -c "python -m compileall scripts"
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#endif()

#catkin_install_python(PROGRAMS scripts/toppra_call.py
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts)

#install(DIRECTORY include/${PROJECT_NAME}/
#  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
