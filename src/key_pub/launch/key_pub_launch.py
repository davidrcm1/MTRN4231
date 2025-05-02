from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='key_pub', 
            executable='key_publisher',  
            name='key_publisher_node',  
            output='screen',
            parameters=[
                
            ],
        ),
    ])
