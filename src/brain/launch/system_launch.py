import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Use the correct package name
xacro_file = os.path.join(
    get_package_share_directory('ur_with_end_effector_description'),  # Updated package name
    'urdf',
    'ur_with_end_effector.xacro'
)

# Parse the xacro file
doc = xacro.parse(open(xacro_file))
xacro.process_doc(doc)
robot_description_config = doc.toxml()

def generate_launch_description():
    use_fake = 'true'  # change this value if needed

    # Set IP addresses based on whether we are using a fake or real robot
    ip_address_fake = 'yyy.yyy.yyy.yyy'
    ip_address_real = '192.168.0.100'

    # UR Control Launch Arguments
    ur_control_launch_args = {
        'ur_type': 'ur5e',
        'robot_ip': ip_address_fake,
        'use_fake_hardware': use_fake,
        'robot_description': robot_description_config,
        'launch_rviz': 'false',
    }

    moveit_launch_args = {
        'ur_type': 'ur5e',
        'launch_rviz': 'true',
    }

    # Update arguments based on whether we are using the fake or real robot
    if use_fake == 'false':
        ur_control_launch_args['robot_ip'] = ip_address_real
        moveit_launch_args['robot_ip'] = ip_address_real
        print("Using Real Robot")
    else:
        ur_control_launch_args['initial_joint_controller'] = 'joint_trajectory_controller'
        moveit_launch_args['use_fake_hardware'] = use_fake
        print("Using Fake Robot")

    # Include UR Control Launch
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'), 'launch', 'ur_control.launch.py'
            ])
        ),
        launch_arguments=ur_control_launch_args.items(),
    )

    # Define MoveIt server launch with a delay
    moveit_launch = TimerAction(
        period=10.0,  # Delay to allow the UR control to start first
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('ur_moveit_config'), 'launch', 'ur_moveit.launch.py'
                    ])
                ),
                launch_arguments=moveit_launch_args.items(),
            ),
        ]
    )

    # Combine all launch descriptions
    launch_description = [
        ur_control_launch,
        moveit_launch,
    ]

    return LaunchDescription(launch_description)
