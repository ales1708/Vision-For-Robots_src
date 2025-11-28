#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageRectifierNode(Node):
    def __init__(self):
        super().__init__('image_rectifier_node')

        # 1. Initialize CV Bridge
        self.bridge = CvBridge()

        # 2. Define Camera Parameters (Hardcoded from your input)
        self.height = 480
        self.width = 640
        
        # Intrinsic Matrix (K)
        self.K = np.array([
            289.11451, 0.0, 347.23664,
            0.0, 289.75319, 235.67429,
            0.0, 0.0, 1.0
        ]).reshape(3, 3)

        # Distortion Coefficients (D)
        self.D = np.array([-0.208848, 0.028006, -0.000705, -0.00082, 0.0])

        # Rectification Matrix (R)
        self.R = np.array([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]).reshape(3, 3)

        # Projection Matrix (P) - 3x4
        # We only need the top-left 3x3 for newCameraMatrix
        self.P = np.array([
            196.89772, 0.0, 342.88724, 0.0,
            0.0, 234.53159, 231.54267, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]).reshape(3, 4)
        
        # 3. Generate Rectification Maps (Do this ONCE at startup)
        self.get_logger().info('Generating rectification maps...')
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=self.K,
            distCoeffs=self.D,
            R=self.R,
            newCameraMatrix=self.P[0:3, 0:3], # Take the 3x3 intrinsic part
            size=(self.width, self.height),
            m1type=cv2.CV_32FC1
        )

        # 4. Create Subscribers and Publishers
        # Change 'camera/image_raw' to your actual input topic name
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            Image,
            'manual_image_rect',
            10
        )
        
        self.get_logger().info('Node initialized. Waiting for images...')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            # 'bgr8' is standard for color images, use 'mono8' if grayscale
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Check if image dimensions match our calibration
            if cv_image.shape[0] != self.height or cv_image.shape[1] != self.width:
                self.get_logger().warn(f"Image size mismatch! Expected {self.width}x{self.height}, got {cv_image.shape[1]}x{cv_image.shape[0]}")
                return

            # Apply the rectification
            rectified_image = cv2.remap(
                cv_image, 
                self.map1, 
                self.map2, 
                interpolation=cv2.INTER_LINEAR
            )

            # Convert OpenCV image back to ROS Image message
            out_msg = self.bridge.cv2_to_imgmsg(rectified_image, encoding='bgr8')
            
            # IMPORTANT: Copy the header from the original message
            # This preserves the timestamp and frame_id (pt_camera_link)
            out_msg.header = msg.header

            # Publish
            self.publisher.publish(out_msg)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'General Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageRectifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()