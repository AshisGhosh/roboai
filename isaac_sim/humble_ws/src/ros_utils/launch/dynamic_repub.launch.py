from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="ros_utils",
                executable="dynamic_repub",
                name="dynamic_repub",
                parameters=[
                    {
                        "topics": [
                            "/pointcloud",
                            "/instance_segmentation",
                            "/agentview/instance_segmentation",
                        ]
                    }
                ],  # Ensure this matches the expected type
            ),
        ]
    )
