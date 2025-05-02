from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    # Define nodes to be launched in order with delays
    dynamic_key_transform_node = Node(
        package='key_transform',
        executable='dynamic_key_transform',
        name='dynamic_key_transform',
        output='screen'
    )

    key_transform_listener_node = Node(
        package='key_transform',
        executable='key_transform_listener',
        name='key_transform_listener',
        output='screen'
    )

    static_camera_to_map_transform_node = Node(
        package='key_transform',
        executable='static_camera_to_map_transform',
        name='static_camera_to_map_transform',
        output='screen'
    )

    # Create TimerActions to enforce order with delays
    dynamic_key_transform_action = TimerAction(
        period=0.0,
        actions=[dynamic_key_transform_node]
    )

    key_transform_listener_action = TimerAction(
        period=2.0,  # Delay to ensure first node is fully up
        actions=[key_transform_listener_node]
    )

    static_camera_to_map_transform_action = TimerAction(
        period=4.0,  # Delay to ensure the second node is fully up
        actions=[static_camera_to_map_transform_node]
    )

    return LaunchDescription([
        dynamic_key_transform_action,
        key_transform_listener_action,
        static_camera_to_map_transform_action,
    ])
