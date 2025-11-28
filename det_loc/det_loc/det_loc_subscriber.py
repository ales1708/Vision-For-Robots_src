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
from .utils.localization_utils import distance_measure, triangulation_3p, triangulation_2p, KalmanFilter2D, get_rotation_rvec
from .utils.camera_panning_utils import ViewTracker
from .utils.rover_detection import rover_detection, overlap_bboxes
import message_filters


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")
        self.calibration = CameraCalibration()
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.br = CvBridge()
        self.last_line = None
        self.pos = []

        self.rgb_frame_main = None
        self.rgb_oak = None
        self.depth_color = None

        self.subscription = self.create_subscription(
            CompressedImage,  # use CompressedImage or Image
            "/image_raw/compressed",
            self.listener_callback,
            10,
        )

        # ---- message_filters subscribers (ROS 2 style) ----
        self.rgb_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            "/oak/rgb/image_raw/compressed",
        )

        self.depth_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            "/oak/stereo/image_raw/compressedDepth",
        )

        # ---- ApproximateTimeSynchronizer ----
        # queue_size: how many msgs to keep in buffer
        # slop: allowed time difference between topics (in seconds)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.oak_callback)

        self.cv_timer = self.create_timer(0.03, self.cv_gui_step)  # ~33 Hz

        # Initialize ViewTracker with image center (assuming 640x480 image)
        self.view_tracker = ViewTracker(self.joint_pub, image_center=(320, 240))

        # Scanning state
        self.is_scanning = True
        self.scanning_initialized = False
        self.scanning_timer = None

        # Start scanning after 1 second delay to allow camera to initialize
        # self.initial_timer = self.create_timer(1.0, self.start_initial_scan)

        self.angular_speed_list = []
        self.detector = apriltag("tagStandard41h12")
        self.tag_size = 0.160 # meters
        self.kf = KalmanFilter2D(dt=0.05)
        self.target = [2.850, 3.0]

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
        # if self.is_scanning and self.scanning_initialized:
        #     current_pan = self.view_tracker.pan_controller.get_pan_position()
        #     self.view_tracker.update_scan_data(detections, current_pan)

        #     # print scanning info in terminal
        #     scan_text = f"SCANNING: Pan={current_pan:.2f}"
        #     self.get_logger().info(scan_text)
        #     self.get_logger().info(f"Detections: {len(detections)}")

        cv2.imshow("detected tags", vis_image)
        self.rgb_frame_main = vis_image

        # Only do localization if scanning is complete
        # if True:
        #     # 5. Measure Distance
        #     # Note: We pass the P_rect_matrix because the image 'frame' is now undistorted.
        #     # scale=1.0 because coordinates are already adjusted to original image size
        #     distance_frame, distances, rvec = distance_measure(
        #         vis_image,
        #         detections,
        #         self.calibration.P_rect_matrix,
        #         self.tag_size,
        #         self.get_logger(),
        #     )

        #     if len(detections) > 1:
        #         if len(detections) > 2:
        #             robot_pos = triangulation_3p(detections, distances)
        #         else:
        #             robot_pos = triangulation_2p(detections, distances)

        #         self.kf.predict()
        #         filtered_pos = self.kf.update(robot_pos)

        #         robot_rotation, robot_rotation_degrees, tag = get_rotation_rvec(
        #             rvec, detections, filtered_pos
        #         )
    
    def oak_callback(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
        # ---------- Decode RGB (JPEG) ----------
        np_rgb = np.frombuffer(rgb_msg.data, np.uint8)
        rgb_frame = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)

        # ---------- Decode Depth (compressedDepth) ----------
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

        # ---------- Convert depth to something viewable ----------
        depth_f = depth_img.astype(np.float32)

        # Optional: inspect min/max once to tune range
        # print("depth min/max:", float(depth_f.min()), float(depth_f.max()))

        # Clip range for visualization (tune for your setup)
        depth_f = np.clip(depth_f, 300, 5000)  # e.g. 0.3–5m or similar

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
            # maybe take mean depth in bbox instead of center pixel?

            depth_value = depth_f[cy - 1, cx - 1] / 1000 # from mm to meters
            depths.append(depth_value)

        #4. Move (bad algorithm, only for testing)
        # self.rover_movement(detections, depths)

        # visualize bboxes on rgb
        np_rgb = np.frombuffer(rgb_msg.data, np.uint8)
        rgb_img = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)
        for det in detections:
            x, y, w, h = det
            cv2.rectangle(rgb_img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
            cv2.circle(rgb_img, (x, y), 5, (255, 0, 0), -1)

        # cv2.imshow("Detected Rovers", rgb_img)
        self.depth_color = depth_color
        self.rgb_oak = rgb_img


    def rover_movement(self, detections, depths, filtered_pos, robot_rotation_degrees):
        min_distance = 0.3 # for obstacle avoidance
        image_center_x = 320    # assuming 640x480 image
        center_zone = 80

        twist = Twist()
        obstacle_detected = False
        steer_direction = 0  # -1: left, 1: right

        # 1. Obstacle avoidance
        for i, det in enumerate(detections):
            x, _, w, _ = det
            depth_val = depths[i]
            bbox_center_x = x + w // 2
            if depth_val < min_distance and abs(bbox_center_x - image_center_x) < center_zone:
                obstacle_detected = True
                steer_direction = 1 if bbox_center_x < image_center_x else -1
                break

        if obstacle_detected:
            twist.linear.x = 0.2 # move a little bit away whilst turning
            twist.angular.z = 0.5 * steer_direction
        else:
            target_pos = self.target
            dx = target_pos[0] - filtered_pos[0]
            dy = target_pos[1] - filtered_pos[1]
            distance_to_target = np.hypot(dx, dy)
            target_angle = np.arctan2(dy, dx)
            angle_error = target_angle - robot_rotation_degrees
            # Normalize angle error to [-pi, pi]?
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

            if distance_to_target < 0.1: # maybe too small/big?
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            elif abs(angle_error) > 0.15: # radians, also maybe too small?
                twist.linear.x = 0.0
                twist.angular.z = 0.5 * np.sign(angle_error)
            else:
                twist.linear.x = 0.4
                twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)

    def cv_gui_step(self):
        # Show main camera with tags
        if self.rgb_frame_main is not None:
            cv2.imshow("detected tags", self.rgb_frame_main)

        # Show OAK RGB
        if self.rgb_oak is not None:
            cv2.imshow("Detected Rovers", self.rgb_oak)

        # Show OAK depth
        if self.depth_color is not None:
            cv2.imshow("Depth Compressed", self.depth_color)

        # 🔑 This single waitKey handles ALL windows
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