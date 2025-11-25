import rclpy
from apriltag import apriltag
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, CameraInfo
from cv_bridge import CvBridge
from .utils.marker_detection_utils import (
    draw_detections,
    multi_scale_marker_detection,
)
from .utils.camera_calibration_utils import CameraCalibration
from .utils.localization_utils import distance_measure, triangulation_3p, triangulation_2p, KalmanFilter2D


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.subscription = self.create_subscription(
            CompressedImage,  # use CompressedImage or Image
            "/image_raw/compressed",
            self.listener_callback,
            10,
        )

        self.calibration = CameraCalibration()
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.br = CvBridge()
        self.last_line = None
        self.initial_camera_timer = self.create_timer(1.0, self.initial_camera_position)

        self.angular_speed_list = []
        self.detector = apriltag("tagStandard41h12")
        self.tag_size = 0.160 # meters
        self.kf = KalmanFilter2D(dt=0.05)
        self.target = [2.850, 3.0]

    def initial_camera_position(self):
        """set camera position to look up"""
        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        msg.position = [0.0, 0.0]  # camera set to look forward
        self.joint_pub.publish(msg)
        self.get_logger().info("Camera turned to preset position.")
        self.destroy_timer(self.initial_camera_timer)

    def listener_callback(self, data):
        """converts recieved images to cv2 images"""

        # 1. Convert to OpenCV
        # raw_frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        np_arr = np.frombuffer(data.data, np.uint8)
        raw_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 2. UNDISTORT the image using provided calibration
        # We use the original K and D to undistort, and map to the new P matrix
        frame = self.calibration.undistort_image(raw_frame)

        # 3. Detect AprilTags using multi-scale detection from marker_detection_utils
        detections = multi_scale_marker_detection(
            frame,
            self.detector,
            scales=[1.5, 2.0, 2.5],
            use_clahe=True,
            use_morphology=False
        )

        # 4. Draw bounding boxes (Visual only)
        vis_image = draw_detections(frame, detections, scale=1.0)
        cv2.imshow("detected tags", vis_image)

        # 5. Measure Distance
        # Note: We pass the P_rect_matrix because the image 'frame' is now undistorted.
        # scale=1.0 because coordinates are already adjusted to original image size
        distance_frame, distances = distance_measure(
            vis_image,
            detections,
            self.P_rect_matrix,
            self.tag_size,
            self.get_logger(),
        )

        if len(detections) > 1:
            if len(detections) > 2:
                robot_pos = triangulation_3p(detections, distances)
            else:
                robot_pos = triangulation_2p(detections, distances)

            self.kf.predict()
            filtered_pos = self.kf.update(robot_pos)

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