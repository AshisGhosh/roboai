#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import rcl_interfaces

from sensor_msgs.msg import Image

class DynamicRepub(Node):
    def __init__(self):
        super().__init__('dynamic_repub')
        self.custom_publishers = {}
        self.initialized_topics = set()

        # Retrieve the list of topics from the parameters
        self.declare_parameter('topics', rcl_interfaces.msg.ParameterValue(type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING_ARRAY),)
        self.topics_to_handle = self.get_parameter('topics').get_parameter_value().string_array_value

        # Check topics periodically to set up publishers and subscribers
        self.timer = self.create_timer(5.0, self.check_and_initialize_topics)  # Check every 5 seconds

    def check_and_initialize_topics(self):
        current_topics = self.get_topic_names_and_types()
        self.get_logger().info(f'Expected topics: {self.topics_to_handle}')
        self.get_logger().info(f'Current topics: {len(current_topics)}')
        for required_topic in self.topics_to_handle:
            if required_topic not in self.initialized_topics:
                for topic_name, types in current_topics:
                    if topic_name == required_topic:
                        self.initialize_pub_sub(topic_name, types[0])
                        break

    def initialize_pub_sub(self, topic_name, type_name):
        if topic_name in self.initialized_topics:
            return  # Already initialized

        # Dynamically import the message type
        msg_type = self.load_message_type(type_name)
        if msg_type is None:
            self.get_logger().error(f'Could not find or load message type for {type_name}')
            return

        sub_qos_profile = QoSProfile(
                                depth=10,
                                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                durability=QoSDurabilityPolicy.VOLATILE
                                )

        repub_topic_name = f"{topic_name}_repub"  # Rename republished topic
        self.custom_publishers[repub_topic_name] = self.create_publisher(msg_type, repub_topic_name, QoSProfile(depth=10))
        self.create_subscription(msg_type, topic_name, lambda msg, repub_topic_name=repub_topic_name: self.repub_callback(msg, repub_topic_name), sub_qos_profile)
        self.initialized_topics.add(topic_name)
        self.get_logger().info(f'Set up republishing from {topic_name} to {repub_topic_name} with type {type_name}')

    def load_message_type(self, type_name):
        # Dynamically load message type from type name
        try:
            package_name, _, message_name = type_name.split('/')
            relative_module = ".msg"
            msg_module = __import__(f"{package_name}{relative_module}", fromlist=[message_name])
            return getattr(msg_module, message_name)
        except (ValueError, ImportError, AttributeError) as e:
            self.get_logger().error(f'Error loading message type {type_name}: {str(e)}')
            return None

    def repub_callback(self, msg, repub_topic_name):
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            msg.header.stamp = self.get_clock().now().to_msg()

        if type(msg) == Image:
            if msg.encoding == '32SC1':
                msg = self._reencode_image(msg)

        self.custom_publishers[repub_topic_name].publish(msg)
    
    def _reencode_image(self, image):
        if image.encoding == '32SC1':
            # Convert the raw data to a numpy array of type int32
            array = np.frombuffer(image.data, dtype=np.int32).reshape(image.height, image.width)

            # Calculate local minimum and maximum values from the array
            min_value = np.min(array)
            max_value = np.max(array)

            # Avoid division by zero if max_value equals min_value
            if max_value == min_value:
                max_value += 1  # Increment max_value to avoid zero division

            # Normalize the array to the range 0-255
            normalized_array = (array - min_value) / (max_value - min_value) * 255
            clipped_array = np.clip(normalized_array, 0, 255)  # Ensure all values are within 0 to 255

            # Convert back to byte format and update the image data
            image.data = clipped_array.astype(np.uint8).tobytes()
            image.encoding = "mono8"
            image.step = image.width  # Number of bytes in a row
        return image

def main(args=None):
    rclpy.init(args=args)
    node = DynamicRepub()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
