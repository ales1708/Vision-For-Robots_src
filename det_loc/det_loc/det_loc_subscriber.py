import rclpy
from apriltag import apriltag
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from .utils.marker_detection_utils import (
    draw_detections,
    multi_scale_marker_detection,
)
from .utils.camera_calibration_utils import CameraCalibration
from .utils.localization_utils import distance_measure, triangulation_3p, triangulation_2p, KalmanFilter2D
from .utils.camera_panning_utils import ViewTracker
from .utils.rover_detection import rover_detection, overlap_bboxes


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.subscription = self.create_subscription(
            CompressedImage,  # use CompressedImage or Image
            "/image_raw/compressed",
            self.listener_callback,
            10,
        )

        # synchornized subscribers?????? hopefully
        # Create subscribers using message_filters
        self.oak_rgb_sub = Subscriber(self, Image, "/oak/rgb/image_raw")
        self.oak_depth_sub = Subscriber(self, Image, "/oak/depth/image_raw")

        # Synchronizer: keeps RGB + depth aligned???
        self.ts = ApproximateTimeSynchronizer(
            [self.oak_rgb_sub, self.oak_depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.oak_callback)


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

        # Create a timer to execute scanning steps
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

        # Log scan results
        self.get_logger().info("="*50)
        self.get_logger().info("SCANNING COMPLETE!")
        self.get_logger().info(f"Total scan positions: {len(self.view_tracker.scan_data)}")

        for i, data in enumerate(self.view_tracker.scan_data):
            self.get_logger().info(
                f"Position {i+1}: Pan={data['pan_position']:.3f}, "
                f"Detections={data['num_detections']:.1f}, "
                f"Center Error={data['center_error']:.2f}"
            )

        # Report best view
        best_view = self.view_tracker.get_best_view()
        if best_view is not None:
            self.get_logger().info("-"*50)
            self.get_logger().info(
                f"BEST VIEW: Pan={best_view['pan_position']:.3f}, "
                f"Detections={best_view['num_detections']:.1f}, "
                f"Center Error={best_view['center_error']:.2f}"
            )
            self.get_logger().info("="*50)

            # Move to best view
            self.view_tracker.move_to_best_view()
            self.get_logger().info("Moved to best view position.")
        else:
            self.get_logger().warn("No valid views found during scan!")
            self.get_logger().info("="*50)

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
            self.view_tracker.update_scan_data(detections, current_pan)

            # print scanning info in terminal
            scan_text = f"SCANNING: Pan={current_pan:.2f}"
            self.get_logger().info(scan_text)
            self.get_logger().info(f"Detections: {len(detections)}")

        cv2.imshow("detected tags", vis_image)

        # Only do localization if scanning is complete
        if not self.is_scanning:
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
    
    def oak_callback(self, rgb, depth):
        # 1. Convert to OpenCV
        rgb = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        frame = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        depth = self.br.imgmsg_to_cv2(depth, desired_encoding="16UC1")

        # 2. Detect obstacles/rovers and remove overlapping bboxes in RGB image
        detections = rover_detection(frame)
        detections = overlap_bboxes(detections)

        # 3. Get distance for each detection w/ depth image
        depths = []
        for det in detections:
            x, y, w, h = det

            # depth ROI in the center of bbox
            cx = x + w // 2
            cy = y + h // 2
            # maybe take mean depth in bbox instead of center pixel?

            depth_value = depth[cy, cx] / 1000 # from mm to meters
            depths.append(depth_value)

        #4. Move (bad algorithm, only for testing)
        self.rover_movement(detections, depths)

        # visualize bboxes on rgb
        # for det in detections:
        #     x, y, w, h = rover
        #     cv2.rectangle(rgb, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        #     cv2.circle(rgb, (x, y), 5, (255, 0, 0), -1)
        # cv2.imshow("Detected Rovers", rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def rover_movement(self, detections, depths, target, filtered_pos, rotation):
        twist = Twist()

        z_error = rotation # maybe add some normalization or whatever here, depends on rotation from fiona

        distance_to_target = np.sqrt((filtered_pos[0] - target[0]) ** 2 + (filtered_pos[1] - target[1]) ** 2)

        if distance_to_target < 0.2:
            speed = 0.1
        else:
            speed = 0.5
        
        if z_error < 0.4:
            rotate = 0.5
        elif z_error > 0.4:
            rotate = -0.5
        elif abs(z_error) <= 0.2:
            rotate = 0.1

        if distance_to_target < 0.05: # risky?
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            twist.linear.x = speed
            twist.angular.z = rotate


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