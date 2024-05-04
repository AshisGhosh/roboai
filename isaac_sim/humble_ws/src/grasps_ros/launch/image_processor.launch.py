from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    node_action = Node(
        package="grasps_ros",
        executable="image_processor",
        name="image_processor",
        output="screen",
        remappings=[
            ("/rgb_image", "/rgb"),
            ("/depth_image", "/depth"),
            ("/camera_info", "/camera_info"),
            ("/grasp_image", "/grasp_image"),
            ("/grasp_markers", "/grasp_markers"),
        ],
    )

    return LaunchDescription([node_action])
