from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
import xacro


def generate_launch_description():
    # Path to xacro file
    description_pkg_share = FindPackageShare("chitrak_description")
    xacro_path = os.path.join(
        description_pkg_share.find("chitrak_description"), "urdf", "chitrak.xacro"
    )

    # Path to RViz config
    bringup_pkg_path = FindPackageShare("chitrak_bringup").find("chitrak_bringup")
    rviz_config_path = os.path.join(bringup_pkg_path, "config", "display.rviz")

    robot_description_content = xacro.process_file(xacro_path).toxml()
    robot_description = {"robot_description": robot_description_content}

    # Robot State Publisher
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )
    # RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
    )
    # Visualize leg trajectories
    viz_leg_trajectories_node = Node(
        package="chitrak_leg_controller",
        executable="viz_leg_trajectories",
        output="screen",
    )

    return LaunchDescription([
        rsp_node,
        rviz_node,
        viz_leg_trajectories_node,
    ])
