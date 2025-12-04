import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.subscription = self.create_subscription(
            Image,  # use CompressedImage or Image
            "/oak/rgb/image_raw",
            self.listener_callback,
            10,
        )

        # self.subscription = self.create_subscription(
        #     Image,
        #     "/oak/rgb/image_raw",
        #     self.listener_callback,
        #     qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        # )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.br = CvBridge()
        self.last_line = None
        # self.initial_camera_timer = self.create_timer(1.0, self.initial_camera_position)
        self.in_rosbag = True
        self.angular_speed_list = []

    def initial_camera_position(self):
        """set camera position to look straight"""

        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        msg.position = [0.0, 0.0]  # 0 degrees and 90 degrees in radians
        self.joint_pub.publish(msg)
        self.get_logger().info("Camera turned to preset position.")
        # self.destroy_timer(self.initial_camera_timer)

    def listener_callback(self, data):
        """Uses the subscribed camera feed to detect lines and follow them"""
        frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")  # for Image
        # np_arr = np.frombuffer(data.data, np.uint8)  # for CompressedImage
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # for CompressedImage
        cv2.imshow("frame", frame)

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
