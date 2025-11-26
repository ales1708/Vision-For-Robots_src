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
from .utils.camera_panning_utils import ViewTracker


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

        # Initialize ViewTracker with image center (assuming 640x480 image)
        self.view_tracker = ViewTracker(self.joint_pub, image_center=(320, 240))

        # Scanning state
        self.is_scanning = True
        self.scanning_initialized = False
        self.scanning_timer = None

        # Start scanning after 1 second delay to allow camera to initialize
        self.initial_timer = self.create_timer(1.0, self.start_initial_scan)

        self.angular_speed_list = []
        self.detector = apriltag("tagStandard41h12")
        self.tag_size = 0.160 # meters
        self.kf = KalmanFilter2D(dt=0.05)
        self.target = [2.850, 3.0]

    def start_initial_scan(self):
        """Initialize the scanning operation"""
        self.get_logger().info("Starting initial scan...")
        self.view_tracker.initial_scanning()
        self.scanning_initialized = True

        # Create a timer to execute scanning steps (slower for better detection)
        self.scanning_timer = self.create_timer(0.1, self.execute_scan_step)

        # Destroy the initialization timer
        self.destroy_timer(self.initial_timer)

    def execute_scan_step(self):
        """Execute one step of the scanning operation"""
        if not self.view_tracker.is_scanning_complete():
            self.view_tracker.pan_controller.scanning_step()
        else:
            # Scanning complete
            self.finish_scanning()

    def finish_scanning(self):
        """Finish the scanning operation and report results"""
        self.is_scanning = False

        # Destroy scanning timer
        if self.scanning_timer is not None:
            self.destroy_timer(self.scanning_timer)
            self.scanning_timer = None

        # Report best view
        best_view = self.view_tracker.get_best_view()
        if best_view is not None:
            meets_req = best_view['num_detections'] >= 2
            status = "✓" if meets_req else "✗"
            self.get_logger().info(
                f"Scan complete | Best view: Pan={best_view['pan_position']:.3f}rad, "
                f"Det={best_view['num_detections']:.1f}, CenterErr={best_view['center_error']:.1f}px {status}"
            )

            # Move to best view and enable tracking
            self.view_tracker.move_to_best_view()
            self.view_tracker.enable_tracking()
            self.get_logger().info("Moved to best view. Tracking enabled.")
        else:
            self.get_logger().warn("Scan complete. No valid views found!")

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

        # If scanning, update scan data
        if self.is_scanning and self.scanning_initialized:
            current_pan = self.view_tracker.pan_controller.get_pan_position()
            frames_accumulated = len(self.view_tracker.current_scan_accumulator)

            self.view_tracker.update_scan_data(detections, current_pan)

            # Only log when we complete a position (reduces spam)
            new_frames = len(self.view_tracker.current_scan_accumulator)
            if new_frames == 0 and frames_accumulated > 0:
                # Position just completed
                num_positions = len(self.view_tracker.scan_data)
                self.get_logger().info(
                    f"Scan position {num_positions} complete | Pan={current_pan:.3f}rad | "
                    f"Avg detections={self.view_tracker.scan_data[-1]['num_detections']:.1f}"
                )

        cv2.imshow("detected tags", vis_image)

        # Only do localization if scanning is complete
        if not self.is_scanning:
            # Apply dynamic tracking to keep tags centered
            if len(detections) >= 1:
                adjusted, error_x, adjustment = self.view_tracker.check_and_adjust_tracking(detections)
                if adjusted:
                    self.get_logger().info(
                        f"Pan adjusted to re-center tags | Error: {error_x:.1f}px | "
                        f"Adjustment: {adjustment:.3f}rad | New pan: {self.view_tracker.pan_controller.get_pan_position():.3f}rad"
                    )

            # 5. Measure Distance
            # Note: We pass the P_rect_matrix because the image 'frame' is now undistorted.
            # scale=1.0 because coordinates are already adjusted to original image size
            distance_frame, distances = distance_measure(
                vis_image,
                detections,
                self.calibration.P_rect_matrix,
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