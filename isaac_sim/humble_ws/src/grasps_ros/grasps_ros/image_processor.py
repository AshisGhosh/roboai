import time
import cv2
from PIL import Image
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose_stamped

import message_filters
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
from roboai_interfaces.action import GetGrasp


from roboai.shared.utils.grasp_client import get_grasp_from_image


def interpolate(p1, p2, num_points=5):
    """Interpolates num_points between p1 and p2, inclusive of p1 and p2."""
    return [
        ((1 - t) * np.array(p1) + t * np.array(p2))
        for t in np.linspace(0, 1, num_points)
    ]


def get_grid_from_box(box):
    sides = []
    for i in range(len(box)):
        p1 = box[i]
        p2 = box[(i + 1) % len(box)]
        if i < len(box) - 1:
            sides.append(interpolate(p1, p2)[:-1])
        else:
            sides.append(interpolate(p1, p2))

    grid = []
    for i in range(len(sides[0])):
        for j in range(len(sides[1])):
            grid.extend(interpolate(sides[0][i], sides[1][j]))

    return grid


def get_angle_from_box(box):
    p1 = box[2]
    p2 = box[3]
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return angle


class ImageProcessor(Node):
    def __init__(self):
        super().__init__("image_processor")
        self.get_logger().info("Image Processor node has been initialized")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_image_ts = None
        rgb_sub = message_filters.Subscriber(self, ROSImage, "/rgb_image")
        depth_sub = message_filters.Subscriber(self, ROSImage, "/depth_image")

        ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 10)
        ts.registerCallback(self.image_callback)

        self.camera_info = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera_info", self.camera_info_callback, 10
        )

        self.publisher = self.create_publisher(ROSImage, "/grasp_image", 10)
        self.bridge = CvBridge()

        self.marker_pub = self.create_publisher(Marker, "/grasp_markers", 10)
        self.grasp_axis_pub = self.create_publisher(Marker, "/grasp_axis_markers", 10)

        self.grasps = None
        self._action_server = ActionServer(
            self,
            GetGrasp,
            "get_grasp",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, rgb_msg, depth_msg):
        # Convert ROS Image message to OpenCV format
        if not self.camera_info:
            self.get_logger().warn("Camera info not available")
            return

        self.last_image_ts = rclpy.time.Time()
        cv_rgb_image = self.bridge.imgmsg_to_cv2(
            rgb_msg, desired_encoding="passthrough"
        )
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        image = Image.fromarray(cv_rgb_image)
        start_time = time.time()
        response = get_grasp_from_image(image)
        self.get_logger().info(f"Time taken to get grasps: {time.time() - start_time}")

        self.handle_response(response, cv_rgb_image, cv_depth_image)

    def handle_response(self, response, cv_rgb_image, cv_depth_image):
        try:
            grasps = response["result"]
            if grasps:
                self.get_logger().info(
                    f"Received grasp poses: {[(grasp['cls_name'], round(grasp['obj'],2)) for grasp in response['result']]}"
                )

                self.publish_grasp_image(grasps, cv_rgb_image)
                self.publish_grasp_markers(grasps, cv_depth_image)

                grasp_dict = self.get_grasp_poses(grasps, cv_depth_image)
                grasp_timestamp = self.get_clock().now().to_msg()
                self.grasps = {
                    "timestamp": grasp_timestamp,
                    "grasps": grasp_dict,
                }
                self.publish_grasp_axis_markers()

        except KeyError as e:
            self.get_logger().warn(
                f"KeyError: Failed to receive valid response or grasp poses: {e}"
            )

    def publish_grasp_image(self, grasps, original_image):
        for grasp in grasps:
            points = np.array(grasp["r_bbox"], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(
                original_image, [points], isClosed=True, color=(0, 255, 0), thickness=2
            )

            # Get the label for the class
            class_label = grasp["cls_name"]
            score = grasp["obj"]
            label_with_score = (
                f"{class_label} ({score:.2f})"  # Formatting score to 2 decimal places
            )
            label_position = (points[0][0][0], points[0][0][1] - 10)
            cv2.putText(
                original_image,
                label_with_score,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        ros_image = self.bridge.cv2_to_imgmsg(original_image, "rgb8")
        self.publisher.publish(ros_image)

    def project_to_3d(self, x, y, depth):
        depth = float(depth)

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        x = (x - cx) * depth / fx
        y = (y - cy) * depth / fy
        return x, y, depth

    def get_grasp_poses(self, grasps, cv_depth_image):
        grasp_poses = {}
        timestamp = self.last_image_ts.to_msg()
        for grasp in grasps:
            grasp_points = grasp["r_bbox"]
            center = np.mean(grasp_points, axis=0)
            average_depth = np.mean(
                [
                    cv_depth_image[int(pt[1]), int(pt[0])]
                    for pt in get_grid_from_box(grasp_points)
                ]
            )

            point_3d = self.project_to_3d(center[0], center[1], average_depth)
            if point_3d:
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = self.camera_info.header.frame_id
                pose_msg.header.stamp = timestamp
                pose_msg.pose.position.x = point_3d[0]
                pose_msg.pose.position.y = point_3d[1]
                pose_msg.pose.position.z = point_3d[2]
                angle = get_angle_from_box(grasp_points)
                pose_msg.pose.orientation.z = np.sin(angle / 2)
                pose_msg.pose.orientation.w = np.cos(angle / 2)

                try:
                    transform = self.tf_buffer.lookup_transform(
                        "world",
                        pose_msg.header.frame_id,
                        timestamp,
                        timeout=rclpy.duration.Duration(seconds=10),
                    )
                    pose_msg = do_transform_pose_stamped(pose_msg, transform)
                    grasp_poses[grasp["cls_name"]] = pose_msg
                except (
                    LookupException,
                    ConnectivityException,
                    ExtrapolationException,
                ) as e:
                    self.get_logger().error(f"Failed to transform point: {str(e)}")

        self.get_logger().info(f"Grasp poses: {len(grasp_poses)}")
        return grasp_poses

    def publish_grasp_markers(self, grasps, cv_depth_image, publish_grid=False):
        if not self.camera_info:
            return

        scale = 0.02
        if publish_grid:
            scale = 0.002

        marker = Marker()
        marker.header.frame_id = self.camera_info.header.frame_id
        marker.header.stamp = self.last_image_ts.to_msg()
        marker.ns = "grasps"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = 1.0
        marker.color.a = 1.0
        marker.points = []
        for grasp in grasps:
            grasp_points = grasp["r_bbox"]

            if publish_grid:
                grasp_points = get_grid_from_box(grasp_points)

            for pt in grasp_points:
                point_3d = self.project_to_3d(
                    pt[0], pt[1], cv_depth_image[int(pt[1]), int(pt[0])]
                )
                if point_3d:
                    marker.points.append(
                        Point(x=point_3d[0], y=point_3d[1], z=point_3d[2])
                    )
        self.marker_pub.publish(marker)

    def publish_grasp_axis_markers(self):
        if not self.grasps:
            return

        marker = Marker()
        marker.header.frame_id = list(self.grasps["grasps"].values())[0].header.frame_id
        marker.header.stamp = self.last_image_ts.to_msg()
        marker.ns = "grasp_axis"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.015
        marker.scale.z = 0.015
        marker.color.b = 1.0
        marker.color.a = 0.5
        # for grasp in self.grasps["grasps"].values():
        grasp = list(self.grasps["grasps"].values())[0]
        # Draw the axis
        marker.pose.position = grasp.pose.position
        marker.pose.orientation = grasp.pose.orientation

        self.grasp_axis_pub.publish(marker)

    def goal_callback(self, goal_request):
        self.get_logger().info("Received goal request")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request")
        self.get_logger().error("Cancel not implemented")
        return CancelResponse.REJECT

    def execute_callback(self, goal_handle):
        self.get_logger().info("Received execute request")

        object_name = goal_handle.request.object_name
        self.get_logger().info(f"Looking for object name: {object_name}")

        time_tolerance = rclpy.time.Duration(seconds=3)
        timeout = 10.0
        while True and timeout > 0:
            self.get_logger().info(
                f"Grasps: {len(self.grasps['grasps'])}, TS: {self.grasps['timestamp']}, now: {self.get_clock().now()}"
            )
            self.get_logger().info(
                f"Time diff: {(self.get_clock().now() - rclpy.time.Time.from_msg(self.grasps['timestamp'])).nanoseconds/1e9}"
            )
            if (
                self.grasps
                and self.get_clock().now()
                - rclpy.time.Time.from_msg(self.grasps["timestamp"])
                < time_tolerance
            ):
                self.get_logger().info(f"Found grasps for object: {object_name}")
                result = GetGrasp.Result(success=True)
                grasp = list(self.grasps["grasps"].values())[0]
                result.grasp = grasp
                goal_handle.succeed()
                return result

            timeout -= 0.1
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().warn(f"Couldn't find grasps for object: {object_name}")
        goal_handle.succeed()
        return GetGrasp.Result(success=False)


def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    try:
        while True:
            rclpy.spin_once(image_processor)
    except KeyboardInterrupt:
        image_processor.get_logger().info("Shutting down")
    image_processor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
