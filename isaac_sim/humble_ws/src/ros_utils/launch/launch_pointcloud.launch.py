from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir


def generate_launch_description():
    # Path to the first launch file
    pointcloud_gen = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [ThisLaunchFileDir(), "/pointcloud_gen.launch.py"]
        )
    )

    # Path to the second launch file
    dynamic_repub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ThisLaunchFileDir(), "/dynamic_repub.launch.py"])
    )

    return LaunchDescription([pointcloud_gen, dynamic_repub])
