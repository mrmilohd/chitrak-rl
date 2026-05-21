from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Path to config file
    description_pkg_share = FindPackageShare('chitrak_description')
    config_path = os.path.join(
        description_pkg_share.find('chitrak_description'), 'config', 'chitrak_params.yaml'
    )

    # Launch the keyboard teleop node
    keyboard_teleop_node = Node(
        package='chitrak_teleop',
        executable='keyboard_teleop',
        name='keyboard_teleop',
    )
    # Launch the body motion planner node
    body_motion_planner_node = Node(
        package='chitrak_gait_planner',
        executable='body_motion_planner',
        name='body_motion_planner',
        parameters=[config_path],
    )
    # Launch the gait scheduler node
    gait_scheduler_node = Node(
        package='chitrak_gait_planner',
        executable='gait_scheduler',
        name='gait_scheduler',
        parameters=[config_path],
    )
    # Launch the bezier gait generator node
    bezier_gait_generator_node = Node(
        package='chitrak_leg_controller',
        executable='bezier_gait_generator',
        name='bezier_gait_generator',
    )

    return LaunchDescription([
        keyboard_teleop_node,
        body_motion_planner_node,
        gait_scheduler_node,
        bezier_gait_generator_node,
    ])
