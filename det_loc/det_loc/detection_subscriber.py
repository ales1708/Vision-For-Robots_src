import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge
import cv2
import numpy as np
from apriltag import apriltag

from det_loc.utils import (
    CameraController,
    ViewTracker,
    TagDetector,
    DistanceMeasurement,
    Triangulation,
    KalmanFilter2D,
    DynamicTagTracker,
)

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = None

    def update(self, error, current_time):
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0

        dt = (current_time - self.prev_time).nanoseconds / 1e9
        if dt <= 0:
            dt = 1e-3
        self.prev_time = current_time

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


class ImageSubscriber(Node):
    def __init__(self, enable_localization=True, enable_navigation=False):
        super().__init__("image_subscriber")

        self.br = CvBridge()

        # Initialize AprilTag detector
        detector = apriltag("tagStandard41h12")

        self.subscription = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.listener_callback,
            10,
        )

        joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)

        # -------- Detection Configuration --------
        detection_config = {
            "scales": [1.5, 2.0, 2.5],
            "use_enhanced_processing": True,
            "duplicate_threshold": 20,
            "draw_detections": True,
        }

        # -------- Camera Control Configuration --------
        camera_config = {
            "pan_min": -1.5,
            "pan_max": 1.5,
            "pan_step": 0.15,
            "center_threshold_px": 60.0,
            "center_correction_gain": 1.2,
        }

        view_config = {
            "center_threshold_px": 60.0,
            "min_tags_for_lock": 2,
            "preferred_tags": 2,
        }

        # Initialize detection and camera control modules
        self.tag_detector = TagDetector(detector, detection_config)
        self.camera = CameraController(joint_pub, self.get_logger(), camera_config)
        self.view_tracker = ViewTracker(self.get_logger(), view_config)

        # -------- Localization Configuration --------
        self.enable_localization = enable_localization
        self.enable_navigation = enable_navigation

        if self.enable_localization:
            # Camera parameters for distance measurement
            camera_params = {
                'fx': 289.11451,
                'fy': 289.75319,
                'cx': 347.23664,
                'cy': 235.67429,
                'distortion': [-0.208848, 0.028006, -0.000705, -0.00082, 0.0]
            }

            self.distance_measure = DistanceMeasurement(camera_params, tag_size=0.288)
            self.kf = KalmanFilter2D(dt=0.05)
            self.robot_pos = None

            # Dynamic tag tracking configuration
            tracker_config = {
                "num_tags_to_track": 2,
                "update_threshold_rad": 0.1,  # ~5.7 degrees
                "camera_offset": 0.0,
                "smoothing_factor": 0.3,
            }
            self.tag_tracker = DynamicTagTracker(
                self.camera,
                config=tracker_config
            )

            if self.enable_navigation:
                self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
                self.target = [4.5, 3.0]  # Default target position [x, y]

        # -------- Scanning State --------
        self.scan_locked = False
        self.startup_wait_sec = 10.0
        self.scan_interval_sec = 0.4
        self.start_time = self.get_clock().now()

        # Current frame data
        self.frame_width = None
        self.detections = []

        # Start scanning timer
        self.scan_timer = self.create_timer(self.scan_interval_sec, self.scanning_step)

    def lock_camera(self, reason):
        """Lock camera to best view and stop scanning."""
        best_pan = self.view_tracker.get_best_pan()
        if best_pan is None:
            self.get_logger().warning("No best view to lock to!")
            return

        self.scan_locked = True
        self.camera.set_pan(best_pan)
        self.destroy_timer(self.scan_timer)

        self.get_logger().info(
            f"{reason} → Locked at pan={self.view_tracker.best_view['pan']:.2f} "
            f"({self.view_tracker.best_view['num_tags']} tags, "
            f"error={self.view_tracker.best_view['center_error']:.1f}px)"
        )

        # Enable dynamic tag tracking if localization is active
        if self.enable_localization and hasattr(self, 'tag_tracker'):
            self.tag_tracker.enable()
            self.get_logger().info("Dynamic tag tracking enabled")

    def scanning_step(self):
        """Main scanning loop: find ideal camera position for tag detection."""
        if self.scan_locked:
            return

        # Wait for startup period
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed < self.startup_wait_sec:
            return

        # Lock if we found ideal view (enough centered tags)
        if self.view_tracker.is_view_ideal():
            self.lock_camera("Found ideal view")
            return

        # Try to center current detections if needed
        if self.camera.should_adjust_to_center(self.detections, self.frame_width):
            self.camera.adjust_pan_to_center(self.detections, self.frame_width)
            return

        # Continue sweeping
        sweep_complete = self.camera.sweep_step()

        # Check if full sweep completed
        if sweep_complete:
            if self.view_tracker.best_view:
                if not self.view_tracker.has_minimum_tags():
                    self.get_logger().warning(
                        f"Sweep complete: only {self.view_tracker.best_view['num_tags']} tag(s) found"
                    )
                self.lock_camera("Sweep complete")
            else:
                self.get_logger().info("Sweep complete: no tags found, centering camera")
                self.camera.center_camera()
                self.destroy_timer(self.scan_timer)


    def listener_callback(self, data):
        """Process incoming camera frames and detect AprilTags."""
        np_arr = np.frombuffer(data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect markers using tag detector
        result, detections = self.tag_detector.detect(frame)

        # Update state
        self.detections = detections
        self.frame_width = frame.shape[1]

        # Track best view during scanning
        if detections and not self.scan_locked:
            center_error = self.camera.calculate_center_error(detections, self.frame_width)
            self.view_tracker.update_best_view(
                detections,
                self.camera.pan_position,
                center_error
            )

        # Localization and navigation (if enabled)
        if self.enable_localization and detections:
            self._process_localization(detections)

        # Display result
        cv2.imshow("Marker Detection", result)
        cv2.waitKey(1)

    def _process_localization(self, detections):
        """Process localization and optionally navigation."""
        # Measure distances to tags
        distances = self.distance_measure.measure_distances(
            detections,
            self.get_logger()
        )

        # Triangulate position
        robot_pos = None
        if len(detections) >= 3:
            robot_pos = Triangulation.triangulate_3p(detections, distances)
        elif len(detections) >= 2:
            robot_pos = Triangulation.triangulate_2p(detections, distances)

        if robot_pos is None:
            return

        # Apply Kalman filter
        self.kf.predict()
        filtered_pos = self.kf.update(robot_pos[:2])

        # Update robot position with filtered values
        self.robot_pos = [filtered_pos[0], filtered_pos[1], robot_pos[2]]

        self.get_logger().info(
            f"Robot position: x={filtered_pos[0]:.2f}, y={filtered_pos[1]:.2f}, "
            f"θ={robot_pos[2]:.2f}"
        )

        # Update dynamic tag tracking (if scan is locked and driving)
        if self.scan_locked and hasattr(self, 'tag_tracker'):
            self.tag_tracker.update(self.robot_pos, self.get_logger())

        # Navigate to target (if enabled)
        if self.enable_navigation and self.scan_locked:
            self._navigate_to_target()

    def _navigate_to_target(self):
        """Simple navigation algorithm to move towards target."""
        if self.robot_pos is None:
            return

        twist = Twist()

        # Calculate angle to target
        dx = self.target[0] - self.robot_pos[0]
        dy = self.target[1] - self.robot_pos[1]
        target_angle = np.arctan2(dy, dx)

        # Calculate heading error (normalized to [-π, π])
        angle_error = target_angle - self.robot_pos[2]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # Calculate distance to target
        distance = np.sqrt(dx**2 + dy**2)

        # Adjust speed based on distance
        if distance < 0.5:
            speed = 0.1
            turning = 0.3
        else:
            speed = 0.3
            turning = 0.5

        # Control logic
        if distance < 0.10:
            # Stop when close enough (10 cm)
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif abs(angle_error) < 0.3:
            # Drive forward if facing target
            twist.linear.x = speed
            twist.angular.z = 0.0
        else:
            # Turn towards target
            twist.linear.x = 0.0
            twist.angular.z = turning if angle_error > 0 else -turning

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
