from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
import xacro


def generate_launch_description():
    # Path to config file
    description_pkg_share = FindPackageShare('chitrak_description')
    config_path = os.path.join(
        description_pkg_share.find('chitrak_description'), 'config', 'chitrak_params.yaml'
    )

    # Path to gazebo xacro and world files
    sim_pkg_share = FindPackageShare('chitrak_sim')
    xacro_path = os.path.join(
        sim_pkg_share.find('chitrak_sim'), 'urdf', 'chitrak_gazebo.xacro'
    )
    world_path = os.path.join(
        sim_pkg_share.find('chitrak_sim'), 'worlds', 'empty_world.sdf'
    )

    # Path to ros_gz_sim launch file
    ros_gz_sim_pkg_path = FindPackageShare('ros_gz_sim')
    gz_sim_launch_path = os.path.join(
        ros_gz_sim_pkg_path.find('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    )

    robot_description_content = xacro.process_file(xacro_path).toxml()
    robot_description = {"robot_description": robot_description_content}

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
    # Launch the IK solver node
    ik_solver_node = Node(
        package='chitrak_leg_controller',
        executable='ik_solver',
        name='ik_solver',
    )
    # Robot State Publisher
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )
    # Launch Gazebo with the specified world
    launch_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            gz_sim_launch_path
        ),
        launch_arguments={
            'gz_args': world_path,
            'on_exit_shutdown': 'true',
        }.items(),
    )
    # Spawn chitrak in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'chitrak',
            '-topic', '/robot_description',
            '-x', '0', '-y', '0', '-z', '0.15',
        ],
    )

    return LaunchDescription([
        keyboard_teleop_node,
        body_motion_planner_node,
        gait_scheduler_node,
        bezier_gait_generator_node,
        ik_solver_node,
        rsp_node,
        launch_gazebo,
        spawn_entity,
    ])
