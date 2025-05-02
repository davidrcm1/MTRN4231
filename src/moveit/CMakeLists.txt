cmake_minimum_required(VERSION 3.8)
project(moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(key_pos_msgs REQUIRED)  # Custom message dependency
find_package(tf2_ros REQUIRED)  # If you need TF for transforms

# Define the executable
add_executable(moveit_node src/moveit.cpp)

# Include directories
target_include_directories(moveit_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Set C++ standard and link dependencies
target_compile_features(moveit_node PUBLIC c_std_99 cxx_std_17)

# Link dependencies
ament_target_dependencies(
  moveit_node
  "moveit_ros_planning_interface"
  "rclcpp"
  "geometry_msgs"
  "key_pos_msgs"
  "tf2_ros"  # Optional: if you need TF
)

# Install the executable
install(TARGETS moveit_node
  DESTINATION lib/${PROJECT_NAME})

# Install the launch directory only
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()
