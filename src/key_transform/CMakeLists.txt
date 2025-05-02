cmake_minimum_required(VERSION 3.8)
project(key_transform)

# Set C++ standard
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(key_pos_msgs REQUIRED)  # Custom message dependency
find_package(rosidl_default_generators REQUIRED)  # For service generation

# Specify the service file and generate service artifacts
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetKeyPositions.srv"
  DEPENDENCIES key_pos_msgs
)

# Add executables
add_executable(dynamic_key_transform src/dynamic_key_transform.cpp)
add_executable(key_transform_listener src/key_transform_listener.cpp)
add_executable(static_camera_to_map_transform src/static_camera_to_map_transform.cpp)

# Link dependencies
ament_target_dependencies(dynamic_key_transform rclcpp geometry_msgs tf2 tf2_ros key_pos_msgs tf2_geometry_msgs)
ament_target_dependencies(key_transform_listener rclcpp geometry_msgs tf2 tf2_ros key_pos_msgs tf2_geometry_msgs)
ament_target_dependencies(static_camera_to_map_transform rclcpp geometry_msgs tf2 tf2_ros key_pos_msgs tf2_geometry_msgs)

# Install executables
install(TARGETS
  dynamic_key_transform
  key_transform_listener
  static_camera_to_map_transform
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
