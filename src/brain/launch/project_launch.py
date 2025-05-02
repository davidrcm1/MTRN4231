from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    # Launch key_pub node as "talker"
    key_pub_node = Node(
        package='key_pub',
        executable='talker',  # Use "talker" instead of "key_publisher"
        name='key_publisher',
        output='screen'
    )

    # Launch dynamic_key_transform node with delay
    dynamic_key_transform_node = TimerAction(
        period=2.0,  # 2-second delay after key_pub_node
        actions=[Node(
            package='key_transform',
            executable='dynamic_key_transform',
            name='dynamic_key_transform',
            output='screen'
        )]
    )

    # Launch key_transform_listener node with delay
    key_transform_listener_node = TimerAction(
        period=4.0,  # 2 seconds after dynamic_key_transform_node
        actions=[Node(
            package='key_transform',
            executable='key_transform_listener',
            name='key_transform_listener',
            output='screen'
        )]
    )

    # Launch static_camera_to_map_transform node with delay
    static_camera_to_map_transform_node = TimerAction(
        period=6.0,  # 2 seconds after key_transform_listener_node
        actions=[Node(
            package='key_transform',
            executable='static_camera_to_map_transform',
            name='static_camera_to_map_transform',
            output='screen'
        )]
    )

    # Launch arduino_comm node with delay
    arduino_node = TimerAction(
        period=8.0,  # 2 seconds after static_camera_to_map_transform_node
        actions=[Node(
            package='arduino_comm',
            executable='arduino_node',
            name='arduino_node',
            output='screen'
        )]
    )

    return LaunchDescription([
        key_pub_node,
        dynamic_key_transform_node,
        key_transform_listener_node,
        static_camera_to_map_transform_node,
        arduino_node
    ])
