import time
import cv2
from PIL import Image
import numpy as np
import rclpy
from rclpy.node import Node

import message_filters
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from roboai.shared.utils.grasp_client import get_grasp_from_image


class ImageProcessor(Node):
    def __init__(self):
        super().__init__("image_processor")
        self.get_logger().info("Image Processor node has been initialized")
        # self.subscription = self.create_subscription(
        #     ROSImage,
        #     '/image_topic',
        #     self.image_callback,
        #     10)

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

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, rgb_msg, depth_msg):
        # Convert ROS Image message to OpenCV format
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
            if len(grasps):
                self.get_logger().info(
                    f"Received grasp poses: {[(grasp['cls_name'], round(grasp['obj'],2)) for grasp in response['result']]}"
                )
                self.publish_grasp_image(grasps, cv_rgb_image)
                self.publish_grasp_markers(grasps, cv_depth_image)
        except Exception as e:
            self.get_logger().warn(
                f"Failed to receive valid response or grasp poses: {e}"
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
        if self.camera_info:
            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]
            x = (x - cx) * depth / fx
            y = (y - cy) * depth / fy
            return x, y, depth
        return None

    def publish_grasp_markers(self, grasps, cv_depth_image):
        if not self.camera_info:
            return

        marker = Marker()
        marker.header.frame_id = self.camera_info.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "grasps"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color.r = 1.0
        marker.color.a = 1.0
        marker.points = []
        for grasp in grasps:
            grasp_points = grasp["r_bbox"]
            for pt in grasp_points:
                point_3d = self.project_to_3d(
                    pt[0], pt[1], cv_depth_image[pt[1], pt[0]]
                )
                if point_3d:
                    marker.points.append(
                        Point(x=point_3d[0], y=point_3d[1], z=point_3d[2])
                    )
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
