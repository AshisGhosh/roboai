import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from roboai_interfaces.msg import ObjectOrder
from roboai_interfaces.srv import SaveObjects
import numpy as np
import cv2
from cv_bridge import CvBridge

class ObjectSorter(Node):
    def __init__(self):
        super().__init__('object_sorter')
        # Parameters
        self.declare_parameter('rgb_topic', '/rgb')
        self.declare_parameter('segmentation_topic', '/instance_segmentation')
        self.declare_parameter('labels_topic', '/segmentation_labels')

        # Subscribers
        self.rgb_subscriber = self.create_subscription(
            Image,
            self.get_parameter('rgb_topic').get_parameter_value().string_value,
            self.rgb_callback,
            10
        )
        self.segmentation_subscriber = self.create_subscription(
            Image,
            self.get_parameter('segmentation_topic').get_parameter_value().string_value,
            self.segmentation_callback,
            10
        )
        self.labels_subscriber = self.create_subscription(
            String,
            self.get_parameter('labels_topic').get_parameter_value().string_value,
            self.labels_callback,
            10
        )

        # Publisher
        self.order_publisher = self.create_publisher(ObjectOrder, 'sorted_objects', 10)

        # Service
        self.save_service = self.create_service(SaveObjects, 'save_objects', self.save_objects_handler)

        # Variables
        self.bridge = CvBridge()
        self.current_rgb_image = None
        self.current_segmentation_image = None
        self.labels = {}

    def rgb_callback(self, msg):
        self.current_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def segmentation_callback(self, msg):
        segmentation_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        self.current_segmentation_image = segmentation_image
        self.process_and_publish_order(segmentation_image)

    def labels_callback(self, msg):
        # self.labels = json.loads(msg.data)
        # labels indexes are 0, 1, 2, 3, etc but pixel values scale to 255
        labels = json.loads(msg.data)
        labels.pop('time_stamp', None)
        # pop where value is BACKGROUND or UNLABBELED
        labels = {k: v for k, v in labels.items() if v not in ['BACKGROUND', 'UNLABELLED']}
        self.labels = {int((i+1)*255/len(labels)): v for i, (k, v) in enumerate(labels.items())}

    def process_and_publish_order(self, segmentation_image):
        object_order = self.sort_objects(segmentation_image)
        if object_order:
            order_msg = ObjectOrder(object_names=object_order)
            self.order_publisher.publish(order_msg)

    def sort_objects(self, segmentation_image):
        unique_objects = np.unique(segmentation_image)
        object_positions = {}
        self.get_logger().info(f"Unique objects: {unique_objects}")
        self.get_logger().info(f"Labels: {self.labels}")
        for obj_id in unique_objects:
            if obj_id == 0 or obj_id not in self.labels:  # Skip background or unknown labels
                self.get_logger().info(f"Skipping object {obj_id}")
                continue
            y, x = np.where(segmentation_image == obj_id)
            min_x = np.min(x)  # Leftmost point
            self.get_logger().info(f"Object {obj_id} at {min_x}")
            object_positions[self.labels[obj_id]] = min_x

        # Sort objects by their leftmost points
        sorted_objects = sorted(object_positions.items(), key=lambda x: x[1])
        return [obj[0] for obj in sorted_objects]

    def save_objects_handler(self, request, response):
        if self.current_rgb_image is None or self.current_segmentation_image is None:
            response.success = False
            response.message = "No data available to save"
            return response

        # Save the RGB image and the object order
        cv2.imwrite('latest_rgb_image.png', self.current_rgb_image)
        order = self.sort_objects(self.current_segmentation_image)
        data = {
            "objects": {
                "count": len(order),
                "names": order
            }
        }
        with open('latest_object_order.json', 'w') as f:
            json.dump(data, f)

        response.success = True
        response.message = "Saved successfully"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ObjectSorter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
