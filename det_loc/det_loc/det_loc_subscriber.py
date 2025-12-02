import rclpy
from apriltag import apriltag
import cv2
import numpy as np
import message_filters
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
from .utils.rover_detection import rover_detection, overlap_bboxes


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.calibration = CameraCalibration()
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.br = CvBridge()
        self.last_line = None

        self.rgb_frame_main = None
        self.rgb_oak = None
        self.depth_color = None

        # pan-tilt camera subscription
        self.subscription = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.listener_callback,
            10,
        )

        # OAK-D rgb subscription
        self.rgb_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            "/oak/rgb/image_raw/compressed",
        )

        # OAK-D depth subscription
        self.depth_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            "/oak/stereo/image_raw/compressedDepth",
        )

        # ApproximateTimeSynchronizer
        # queue_size: how many msgs to keep in buffer
        # slop: allowed time difference between topics (in seconds)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.oak_callback)

        # Initialize ViewTracker with image center (assuming 640x480 image)
        self.view_tracker = ViewTracker(self.joint_pub, image_center=(320, 240))
        self.cv_timer = self.create_timer(0.03, self.cv_gui_step)  # ~33 Hz

        # Scanning state
        self.is_scanning = True
        self.scanning_initialized = False
        self.scanning_timer = None
        self.initial_timer = self.create_timer(1.0, self.start_initial_scan)
        self.first_scan = True

        # Detection loss tracking - only rescan after consecutive failures
        self.no_detection_count = 0
        self.no_detection_threshold = 15

        # Pan drift threshold - trigger rescan if camera drifts too far from best view position
        self.pan_drift_threshold = 1.0  # radians (~57 degrees) of drift from reference
        self.reference_pan_position = 0.0  # Set after each scan completes

        self.angular_speed_list = []
        self.detector = apriltag("tagStandard41h12")
        self.tag_size = 0.160  # meters
        self.kf = KalmanFilter2D(dt=0.05)
        self.target = [1.3, 3.0]  # penalty dot
        self.filtered_pos = [0.0, 0.0]
        self.rotation_degrees = 0.0

        self.detections = []
        self.depths = []

    def publish_forward_speed(self, linear_x, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def start_initial_scan(self):
        """Initialize the full scanning operation (used for first scan)."""
        self.get_logger().info("Starting initial full scan...")

        self.is_scanning = True
        self.scanning_initialized = True
        self.no_detection_count = 0  # Reset detection loss counter

        self.view_tracker.initial_scanning()
        self.scanning_timer = self.create_timer(0.1, self.execute_scan_step)

        if self.initial_timer:
            self.destroy_timer(self.initial_timer)
            self.initial_timer = None

    def start_smart_rescan(self):
        """Start smart rescan - moves from current position towards center,
        stops when 2+ tags are found."""
        self.get_logger().info(
            f"Starting smart rescan from pan={self.view_tracker.pan_controller.get_pan_position():.2f}rad"
        )

        self.is_scanning = True
        self.scanning_initialized = True
        self.no_detection_count = 0

        self.view_tracker.start_smart_rescan()
        self.scanning_timer = self.create_timer(0.1, self.execute_smart_rescan_step)

    def execute_scan_step(self):
        """Execute a single step of the full scanning operation."""
        if self.view_tracker.is_scanning_complete():
            self.finish_scanning()
            return

        self.view_tracker.pan_controller.scanning_step()

    def execute_smart_rescan_step(self):
        """Execute a single step of smart rescan.
        Keeps panning back and forth until tags are found.
        """
        self.view_tracker.smart_rescan_step()


    def finish_scanning(self):
        """Finish the full scanning operation and report results."""
        self.is_scanning = False
        self.scanning_initialized = False
        self.first_scan = False

        if self.scanning_timer:
            self.destroy_timer(self.scanning_timer)
            self.scanning_timer = None

        best_view = self.view_tracker.get_best_view()
        if best_view:
            self.get_logger().info(
                f"Scan complete | Best view: "
                f"Pan={best_view['pan_position']:.3f}rad, "
                f"Det={best_view['num_detections']:.1f}, "
                f"CenterErr={best_view['center_error']:.1f}px, "
                f"Score={self.view_tracker.best_view_score:.3f}"
            )

            # Move to best view and enable tracking
            self.view_tracker.move_to_best_view()
            self.view_tracker.enable_tracking()

            # Store reference position for drift detection
            self.reference_pan_position = best_view['pan_position']
            self.get_logger().info(
                f"Moved to best view (ref={self.reference_pan_position:.3f}rad). Tracking enabled."
            )
        else:
            # No valid views found - listener_callback will trigger a new scan
            self.get_logger().warn("Scan complete. No valid views found! Will rescan...")

    def finish_smart_rescan(self, found_tags=False):
        """Finish the smart rescan operation."""
        self.is_scanning = False
        self.scanning_initialized = False

        if self.scanning_timer:
            self.destroy_timer(self.scanning_timer)
            self.scanning_timer = None

        if found_tags:
            # Stay at current position and enable tracking
            current_pan = self.view_tracker.pan_controller.get_pan_position()
            self.reference_pan_position = current_pan
            self.view_tracker.enable_tracking()
            self.get_logger().info(
                f"Smart rescan found tags at pan={current_pan:.3f}rad. Tracking enabled."
            )
        else:
            # Didn't find tags - will trigger full rescan from listener_callback
            self.get_logger().warn("Smart rescan failed to find tags. Will trigger full rescan...")

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
            scales=[1.5, 2.5, 3.5, 4.5],
            use_clahe=True,
            use_morphology=False,
        )

        # 4. Draw bounding boxes
        vis_image = draw_detections(frame, detections, scale=1.0)
        self.rgb_frame_main = vis_image

        if self.is_scanning and self.scanning_initialized:
            current_pan = self.view_tracker.pan_controller.get_pan_position()
            
            # During smart rescan: stop immediately if we find 2+ tags
            if self.view_tracker.is_smart_rescan_active() and len(detections) >= 2:
                self.get_logger().info(
                    f"Smart rescan found {len(detections)} tags at pan={current_pan:.2f}rad - stopping"
                )
                self.view_tracker.finish_smart_rescan(found_tags=True)
                self.finish_smart_rescan(found_tags=True)
                return  # Exit early, next frame will use tracking
            
            # For full scan: update scan data
            if not self.view_tracker.is_smart_rescan_active():
                self.view_tracker.update_scan_data(detections, current_pan)

        # 5. Perform localization with new dictionary-based system
        if len(detections) >= 2 and not self.is_scanning:
            # Reset no-detection counter when we have good detections
            self.no_detection_count = 0

            # Dynamic tracking: adjust pan to keep tags centered
            adjusted, error_x, adjustment = self.view_tracker.check_and_adjust_tracking(detections)
            if adjusted:
                self.get_logger().info(
                    f"Tracking adjustment: error={error_x:.1f}px, adj={adjustment:.4f}rad"
                )

            # Check if pan has drifted too far from reference position - trigger smart reorientation
            current_pan = self.view_tracker.pan_controller.get_pan_position()
            drift_from_reference = abs(current_pan - self.reference_pan_position)
            if drift_from_reference > self.pan_drift_threshold:
                self.get_logger().info("Pan drift exceeded threshold - triggering smart rescan")
                self.view_tracker.disable_tracking()
                self.start_smart_rescan()
                return  # Exit early

            # Measure distances and get detailed detection info
            distance_frame, detections_info, rvecs = distance_measure(
                vis_image,
                detections,
                self.calibration.P_rect_matrix,
                self.tag_size,
                self.get_logger(),
            )

            # 6. Filter out invalid detections
            valid_detections = [
                d for d in detections_info
                if d['distance'] is not None and d['tvec'] is not None
            ]

            if len(valid_detections) >= 2:
                # 7. Perform triangulation with rotation
                if len(valid_detections) >= 3:
                    robot_pos, robot_rotation = triangulation_3p(valid_detections)
                else:
                    robot_pos, robot_rotation = triangulation_2p(valid_detections[:2])

                # 8. Apply Kalman filter for position smoothing
                self.kf.predict()
                filtered_pos = self.kf.update(robot_pos)
                self.filtered_pos = filtered_pos

                # 9. Convert rotation to degrees for display
                robot_rotation_degrees = np.degrees(robot_rotation)
                self.rotation_degrees = robot_rotation_degrees

                # 10. Log results
                self.get_logger().info(
                    f"Position: ({filtered_pos[0]:.3f}, {filtered_pos[1]:.3f}) | "
                    f"Rotation: {robot_rotation_degrees:.1f}° | "
                    f"Tags: {[d['id'] for d in valid_detections[:2]]}"
                )

                self.rover_movement()
            else: # not enough valid tags found
                self.get_logger().info("Not enough valid tags found")
                # self.publish_forward_speed(0.2)
        else: # not enough tags found or currently scanning
            if self.is_scanning and self.scanning_initialized: # continuing to scan
                if not self.first_scan:
                    # self.publish_forward_speed(0.05)
                    pass
                else:
                    self.publish_forward_speed(0.0)
            else: # not currently scanning - check if we should start one
                self.no_detection_count += 1

                if self.no_detection_count >= self.no_detection_threshold:
                    self.no_detection_count = 0  # Reset counter
                    
                    if self.first_scan:
                        # First scan - do full sweep
                        self.get_logger().info(
                            f"No detections for {self.no_detection_threshold} frames - starting full scan"
                        )
                        self.start_initial_scan()
                    else:
                        # After first scan - use smart rescan (move towards center)
                        self.get_logger().info(
                            f"No detections for {self.no_detection_threshold} frames - starting smart rescan"
                        )
                        self.start_smart_rescan()


    def oak_callback(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
        """Converts OAK-D images to cv2 images and performs rover detection."""
        # 1. Obtain rgb and depth images from OAK-D
        # Decode RGB (JPEG)
        np_rgb = np.frombuffer(rgb_msg.data, np.uint8)
        rgb_frame = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)

        # Decode Depth (compressedDepth)
        # format example: "16UC1; compressedDepth"
        depth_fmt, compr_type = depth_msg.format.split(";")
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()

        # 12-byte header before the PNG image
        depth_header_size = 12
        raw_depth_data = depth_msg.data[depth_header_size:]

        np_depth = np.frombuffer(raw_depth_data, np.uint8)
        depth_img = cv2.imdecode(np_depth, cv2.IMREAD_UNCHANGED)

        # If depth image has 3 channels for some reason, take the first
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]

        # Convert depth to something viewable
        depth_f = depth_img.astype(np.float32)
        depth_f = np.clip(depth_f, 300, 5000)  # 0.3–5m

        depth_vis = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 2. Detect obstacles/rovers and remove overlapping bboxes in RGB image
        detections = rover_detection(rgb_frame)
        detections = overlap_bboxes(detections)

        # 3. Get distance for each detection w/ depth image
        depths = []
        for det in detections:
            x, y, w, h = det

            # depth ROI in the center of bbox
            cx = x + w // 2
            cy = y + h // 2

            depth_value = depth_f[cy - 1, cx - 1] / 1000 # from mm to meters
            depths.append(depth_value)

        # visualize bboxes on rgb (optional)
        np_rgb = np.frombuffer(rgb_msg.data, np.uint8)
        rgb_img = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)
        for det in detections:
            x, y, w, h = det
            cv2.rectangle(rgb_img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
            cv2.circle(rgb_img, (x, y), 5, (255, 0, 0), -1)

        self.depth_color = depth_color
        self.rgb_oak = rgb_img
        self.detections = detections
        self.depths = depths

    def rover_movement(self):
        """
        Control rover movement towards target.

        Args:
            target: [x, y] target position
            filtered_pos: [x, y] current robot position
            rotation_degrees: Current robot rotation in degrees
        """
        twist = Twist()

        min_distance = 0.3 # for obstacle avoidance
        image_center_x = 640    # assuming 1280xsomething image
        center_zone = 300
        obstacle_detected = False
        steer_direction = 0  # -1: left, 1: right

        filtered_pos = self.filtered_pos
        rotation_degrees = self.rotation_degrees
        target = self.target

        # Calculate angle to target
        target_angle = np.arctan2(
            target[1] - filtered_pos[1],
            target[0] - filtered_pos[0]
        )
        target_angle_degrees = np.degrees(target_angle)

        # Calculate turn needed
        turn_to_target = rotation_degrees - target_angle_degrees

        # Normalize to [-180, 180] degrees
        turn_to_target = (turn_to_target + 180) % 360 - 180

        # Calculate distance to target
        distance_to_target = np.sqrt(
            (filtered_pos[0] - target[0]) ** 2 +
            (filtered_pos[1] - target[1]) ** 2
        )
        for i, det in enumerate(self.detections):
            x, _, w, _ = det
            depth_val = self.depths[i]
            bbox_center_x = x + w // 2
            if depth_val < min_distance and abs(bbox_center_x - image_center_x) < center_zone:
                obstacle_detected = True
                steer_direction = 1 if bbox_center_x < image_center_x else -1
                break

        if obstacle_detected:
            # move a little bit to the side to avoid obstacle
            twist.linear.x = 0.2
            twist.angular.z = 0.5 * steer_direction
        else:
            # Determine speeds based on distance
            if distance_to_target < 0.2:
                speed = 0.1
            else:
                speed = 0.3

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
            if distance_to_target < 0.15:
                # Reached target
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            elif rotate != 0:
                # Need to turn first
                twist.linear.x = speed * 0.5 # slower when turning
                twist.angular.z = rotate
            else:
                # Move forward
                twist.linear.x = speed
                twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)

    def cv_gui_step(self):
        # Show main camera with tags
        if self.rgb_frame_main is not None:
            cv2.imshow("detected tags", self.rgb_frame_main)

        # # Show OAK RGB
        # if self.rgb_oak is not None:
        #     cv2.imshow("Detected Rovers", self.rgb_oak)

        # # Show OAK depth
        # if self.depth_color is not None:
        #     cv2.imshow("Depth Compressed", self.depth_color)

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