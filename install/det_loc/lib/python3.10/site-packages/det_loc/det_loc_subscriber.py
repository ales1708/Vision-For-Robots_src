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
from .utils.localization_utils import (
    distance_measure,
    triangulation_3p,
    triangulation_2p,
    KalmanFilter2D,
)
from .utils.camera_panning_utils import ViewTracker


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.subscription = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.listener_callback,
            10,
        )

        self.calibration = CameraCalibration()
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.br = CvBridge()
        self.last_line = None

        # Initialize ViewTracker with image center (assuming 640x480 image)
        self.view_tracker = ViewTracker(self.joint_pub, image_center=(320, 240))

        # Scanning state
        self.is_scanning = True
        self.scanning_initialized = False
        self.scanning_timer = None

        self.angular_speed_list = []
        self.detector = apriltag("tagStandard41h12")
        self.tag_size = 0.160  # meters
        self.kf = KalmanFilter2D(dt=0.05)
        self.target = [1.3, 3.0]  # penalty dot

    def listener_callback(self, data):
        """Converts received images to cv2 images and performs localization"""

        # 1. Convert to OpenCV
        np_arr = np.frombuffer(data.data, np.uint8)
        raw_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 2. Undistort the image using provided calibration
        frame = self.calibration.undistort_image(raw_frame)

        # 3. Detect AprilTags using multi-scale detection
        detections = multi_scale_marker_detection(
            frame,
            self.detector,
            scales=[1.5, 2.0, 2.5],
            use_clahe=True,
            use_morphology=False,
        )

        # 4. Draw bounding boxes (Visual only)
        vis_image = draw_detections(frame, detections, scale=1.0)
        cv2.imshow("detected tags", vis_image)

        # 5. Perform localization with new dictionary-based system
        if len(detections) >= 2:
            # Measure distances and get detailed detection info
            distance_frame, detections_info, rvecs = distance_measure(
                vis_image,
                detections,
                self.calibration.P_rect_matrix,
                self.tag_size,
                self.get_logger(),
            )

            # Filter out invalid detections
            valid_detections = [
                d for d in detections_info 
                if d['distance'] is not None and d['tvec'] is not None
            ]

            if len(valid_detections) >= 2:
                # Perform triangulation with rotation
                if len(valid_detections) >= 3:
                    robot_pos, robot_rotation = triangulation_3p(valid_detections)
                else:
                    robot_pos, robot_rotation = triangulation_2p(valid_detections[:2])

                # Apply Kalman filter for position smoothing
                self.kf.predict()
                filtered_pos = self.kf.update(robot_pos)
                
                # Convert rotation to degrees for display
                robot_rotation_degrees = np.degrees(robot_rotation)

                # Log results
                self.get_logger().info(
                    f"Position: ({filtered_pos[0]:.3f}, {filtered_pos[1]:.3f}) | "
                    f"Rotation: {robot_rotation_degrees:.1f}° | "
                    f"Tags: {[d['id'] for d in valid_detections[:2]]}"
                )

                # Uncomment to enable rover movement
                # self.rover_movement(self.target, filtered_pos, robot_rotation_degrees)

        cv2.waitKey(1)

    def rover_movement(self, target, filtered_pos, rotation_degrees):
        """
        Control rover movement towards target.
        
        Args:
            target: [x, y] target position
            filtered_pos: [x, y] current robot position
            rotation_degrees: Current robot rotation in degrees
        """
        twist = Twist()

        # Calculate angle to target
        target_angle = np.arctan2(
            target[1] - filtered_pos[1], 
            target[0] - filtered_pos[0]
        )
        target_angle_degrees = np.degrees(target_angle)
        
        # Calculate turn needed
        turn_to_target = rotation_degrees - target_angle_degrees
        
        # Normalize to [-180, 180]
        turn_to_target = (turn_to_target + 180) % 360 - 180
        
        # Calculate distance to target
        distance_to_target = np.sqrt(
            (filtered_pos[0] - target[0]) ** 2 + 
            (filtered_pos[1] - target[1]) ** 2
        )

        # Determine speeds based on distance
        if distance_to_target < 0.2:
            speed = 0.1
        else:
            speed = 0.5

        # Determine rotation based on angle error
        if abs(turn_to_target) > 30:
            rotate = 0.5 if turn_to_target < 0 else -0.5
        elif abs(turn_to_target) > 10:
            rotate = 0.3 if turn_to_target < 0 else -0.3
        else:
            rotate = 0.0

        self.get_logger().info(
            f"Target angle: {target_angle_degrees:.1f}°, Turn needed: {turn_to_target:.1f}°, "
            f"Distance: {distance_to_target:.3f}m"
        )

        # Set movement commands
        if distance_to_target < 0.05:
            # Reached target
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif rotate != 0:
            # Need to turn first
            twist.linear.x = 0.0
            twist.angular.z = rotate
        else:
            # Move forward
            twist.linear.x = speed
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)


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