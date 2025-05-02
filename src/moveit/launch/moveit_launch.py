# moveit_launch.py
 
from launch import LaunchDescription
from launch_ros.actions import Node
 
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='moveit',
            executable='moveit_node',
            name='moveit_node_instance_1',
            output='screen'
        ),
    ])