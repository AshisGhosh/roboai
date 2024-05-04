from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="data_gen",
                executable="object_sorter",
                name="object_sorter",
                parameters=[
                    {"rgb_topic": "/agentview/rgb"},
                    {"segmentation_topic": "/agentview/instance_segmentation_repub"},
                    {"labels_topic": "/agentview/semantic_labels"},
                ],
            )
        ]
    )
