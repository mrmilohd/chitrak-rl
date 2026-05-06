from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Path to config file
    description_pkg_share = FindPackageShare('chitrak_description')
    config_path = os.path.join(description_pkg_share.find('chitrak_description'), 'config', 'chitrak_params.yaml')

    # Launch the keyboard teleop node
    keyboard_teleop_node = Node(
        package='chitrak_teleop',
        executable='keyboard_teleop',
        name='keyboard_teleop',
    )
    # Launch the gait controller node
    gait_controller_node = Node(
        package='chitrak_gait_controller',
        executable='gait_controller',
        name='gait_controller',
        parameters=[config_path],
    )

    return LaunchDescription([
        keyboard_teleop_node,
        gait_controller_node,
    ])
