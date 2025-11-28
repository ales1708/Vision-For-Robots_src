import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, CameraInfo
from cv_bridge import CvBridge

class CameraCalibration:
    def __init__(self):
        self.K_matrix = np.array([
            [286.9896545092208, 0.0, 311.7114840273407],
            [0.0, 290.8395992360502, 249.9287049631703],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.D_coeffs = np.array(
            [-0.17995587161461585, 0.020688274841999105, -0.005297672531455161, 0.003378882156848116, 0.0],
            dtype=np.float32
        )

        self.P_rect_matrix = np.array([
            [190.74984649646845, 0.0, 318.0141593176815],
            [0.0, 247.35103262891005, 248.37293105876694],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def undistort_image(self, image):
        return cv2.undistort(image, self.K_matrix, self.D_coeffs, None, self.P_rect_matrix)