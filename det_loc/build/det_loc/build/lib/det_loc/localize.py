import rclpy
from apriltag import apriltag
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, CameraInfo
from cv_bridge import CvBridge
import time
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.image_stream = self.create_subscription(
            Image,  # use CompressedImage or Image
            "camera/image_raw",
            self.listener_callback,
            10,
        )

        self.br = CvBridge()

        self.detector = apriltag("tagStandard41h12")
        self.tag_size = 0.160   

    def listener_callback(self, data):
        """converts recieved images to cv2 images"""

        frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")  # for Image
        cv2.imshow("Image Window", frame)
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
