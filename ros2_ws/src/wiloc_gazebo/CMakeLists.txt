cmake_minimum_required(VERSION 3.8)
project(wiloc_gazebo)

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(wiloc_key_forwarder SHARED src/key_forwarder.cpp)
target_compile_features(wiloc_key_forwarder PUBLIC cxx_std_17)
ament_target_dependencies(wiloc_key_forwarder gazebo_dev gazebo_ros rclcpp std_msgs)

install(TARGETS wiloc_key_forwarder LIBRARY DESTINATION lib)

ament_package()
