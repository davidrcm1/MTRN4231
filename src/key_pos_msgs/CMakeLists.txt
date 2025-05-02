cmake_minimum_required(VERSION 3.8)
project(key_pos_msgs)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeyPosition.msg"
  "msg/KeyPositionArray.msg"
  "msg/MoveItCoords.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Export dependencies for other packages
ament_export_dependencies(rosidl_default_runtime std_msgs geometry_msgs)

ament_package()
