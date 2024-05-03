from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    container = ComposableNodeContainer(
        namespace='',
        name='depth_image_proc_container',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::PointCloudXyzrgbNode',
                name='point_cloud_xyzrgb_node',
                remappings=[
                    ('rgb/camera_info', '/camera_info'),
                    ('rgb/image_rect_color', 'rgb'),
                    ('depth_registered/image_rect', 'depth'),
                    ('points', 'pointcloud'),
                ],
                parameters=[{
                    'use_sim_time': False,
                    'queue_size': 10,

                    'qos_overrides./parameter_events.publisher.durability': 'transient_local'
                }]
            ),
        ]
    )

    return LaunchDescription([container])