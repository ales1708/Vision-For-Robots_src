import rclpy
from apriltag import apriltag
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, CameraInfo
from cv_bridge import CvBridge


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.subscription = self.create_subscription(
            CompressedImage,  # use CompressedImage or Image
            "/image_raw/compressed",
            self.listener_callback,
            10,
        )

        # Manual Calibration Parameters (Provided by user)
        # K: Original Camera Matrix
        self.K_matrix = np.array([
            [286.9896545092208, 0.0, 311.7114840273407],
            [0.0, 290.8395992360502, 249.9287049631703],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # D: Distortion Coefficients
        self.D_coeffs = np.array(
            [-0.17995587161461585, 0.020688274841999105, -0.005297672531455161, 0.003378882156848116, 0.0],
            dtype=np.float32
        )

        # P: Projection/New Camera Matrix (Used after undistortion)
        # Taking the 3x3 part of the 3x4 P matrix provided
        self.P_rect_matrix = np.array([
            [190.74984649646845, 0.0, 318.0141593176815],
            [0.0, 247.35103262891005, 248.37293105876694],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

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
        frame = cv2.undistort(raw_frame, self.K_matrix, self.D_coeffs, None, self.P_rect_matrix)

        # 3. Preprocess (Scale Up + Grayscale)
        processed_frame, scale = preprocess_image(frame)

        # 4. Detect AprilTags
        detections = self.detector.detect(processed_frame)

        # 5. Draw bounding boxes (Visual only)
        vis_image = draw_detections(frame, detections, scale=scale)
        cv2.imshow("detected tags", vis_image)

        # 6. Measure Distance
        # Note: We pass the P_rect_matrix because the image 'frame' is now undistorted.
        distance_frame, distances = distance_measure(
            vis_image,
            detections,
            self.P_rect_matrix, 
            self.tag_size,
            scale,
            self.get_logger(),
        )

        if len(detections) > 1:
            if len(detections) > 2:
                robot_pos = triangulation_3p(detections, distances)
            else:
                robot_pos = triangulation_2p(detections, distances)

            # print(f"robot_pos: {robot_pos}")

            # self.kf.predict()
            # filtered_pos = self.kf.update(robot_pos)

        cv2.waitKey(1)


def preprocess_image(color_image):
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    scale = 2.0
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image, scale


def draw_detections(color_image, detections, scale=2.0):
    """
    Draw bounding boxes and tag IDs on the color image.
    """
    vis_image = color_image.copy()

    for detection in detections:
        # Get corner coordinates and scale them back to original image size
        corners = detection["lb-rb-rt-lt"]
        corners = [(int(x / scale), int(y / scale)) for x, y in corners]

        # Draw bounding box
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i + 1) % 4]
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)

        center = (
            int(detection["center"][0] / scale),
            int(detection["center"][1] / scale),
        )

        cv2.circle(vis_image, center, 5, (0, 0, 255), -1)

        tag_id = detection["id"]
        cv2.putText(
            vis_image,
            f"ID: {tag_id}",
            (center[0] - 20, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    return vis_image


def distance_measure(frame, results, camera_matrix, tag_size, scale, logger):
    """
    Measures the distance to an apriltag.
    Uses the Undistorted Camera Matrix (P) and 0 distortion.
    """

    if len(results) == 0:
        return frame, None

    half = tag_size / 2.0

    # Counter-Clockwise starting Bottom-Left
    object_points = np.array(
        [
            [-half, -half, 0],  # Bottom-Left
            [half, -half, 0],   # Bottom-Right
            [half, half, 0],    # Top-Right
            [-half, half, 0],   # Top-Left
        ],
        dtype=np.float32,
    )

    # Since the image passed in is already undistorted, we assume 0 distortion for solvePnP
    dist_coeffs = np.zeros((4, 1))

    distances = []
    for det in results:
        # Get corners from detection (which was done on scaled image)
        corners_raw = np.array(det["lb-rb-rt-lt"], dtype=np.float32)
        
        # CRITICAL: Scale corners back to the original resolution 
        # because our camera_matrix corresponds to the original 640x480 resolution.
        corners = corners_raw / scale

        if corners.shape == (4, 2):
            corners = np.ascontiguousarray(corners)

            success, rvec, tvec = cv2.solvePnP(
                object_points, corners, camera_matrix, dist_coeffs
            )

            if success:
                dist = float(np.linalg.norm(tvec))
                distances.append(dist)
            else:
                distances.append(None)
        else:
            distances.append(None)

    # Logging
    for i in range(len(distances)):
        if distances[i] is not None:
            # Optional: throttle this log to avoid spamming console
            pass
            logger.info(f"AprilTag {results[i]['id']} distance: {distances[i]:.3f} m")

    return frame, distances


def triangulation_3p(detections, distances):
    """Performs triangulation to compute the position of the Robot with 3 points."""
    # get intersections between all three points
    [ab_x, ab_y] = triangulation_2p(detections[:2], distances[:2])
    [ac_x, ac_y] = triangulation_2p(
        [detections[0], detections[2]], [distances[0], distances[2]]
    )
    [bc_x, bc_y] = triangulation_2p(detections[1:3], distances[1:3])

    # get mean of intersections
    x = np.mean(np.array([ab_x, ac_x, bc_x]))
    y = np.mean(np.array([ab_y, ac_y, bc_y]))

    return [x, y]


def triangulation_2p(detections, distances):
    """Performs triangulation to compute the position of the Robot with 2 points."""
    # locations of tags
    april_tags = {
        "tag1": [9.0, 3.0],
        "tag2": [7.350, 0.0],
        "tag3": [7.350, 6.0],
        "tag4": [4.5, 0.0],
        "tag5": [4.5, 6.0],
        "tag6": [2.644, -0.28],
        "tag7": [2.644, 6.78],
        "tag8": [0, 0.144],
        "tag9": [0.0, 3.0],
        "tag10": [0.0, 5.866],
    }

    # get detected tags
    a = detections[0]["id"]
    b = detections[1]["id"]
    
    # Verify tags exist in dict
    if f"tag{a}" not in april_tags or f"tag{b}" not in april_tags:
        print(f"Tag ID {a} or {b} not found in map")
        return [0.0, 0.0]

    [ax, ay] = april_tags[f"tag{a}"]
    [bx, by] = april_tags[f"tag{b}"]

    [a_d, b_d] = distances

    if a_d is None or b_d is None:
        return [0.0, 0.0]

    # get euclidean distance between a and b
    d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

    # Triangle inequality check (with slight tolerance)
    if d > (a_d + b_d) * 1.05: 
        print("circles dont intersect")

    # get triangle geometry
    try:
        z = (a_d**2 - b_d**2 + d**2) / (2 * d)
        h = np.sqrt(max(0, a_d**2 - z**2)) # prevent sqrt of negative
        cx = ax + z * (bx - ax) / d
        cy = ay + z * (by - ay) / d

        # get possible robot coordinates
        qx = cx + h * (by - ay) / d
        qy = cy - h * (bx - ax) / d
        px = cx - h * (by - ay) / d
        py = cy + h * (bx - ax) / d

        # remove the extraneous solution that lies outside the field.
        if qx < 0 or qx > 9 or qy < 0 or qy > 6:
             robot_pos = [px, py]
        else:
             robot_pos = [qx, qy]
             
    except Exception as e:
        print(f"Triangulation error: {e}")
        return [0.0, 0.0]

    return robot_pos


# based on: https://www.geeksforgeeks.org/python/kalman-filter-in-python/
class KalmanFilter2D:
    def __init__(self, dt=0.05):
        # state vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # state transition matrix
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # observation matrix: we measure only x,y
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

        # covariance matrix
        self.P = np.eye(4)

        # process noise
        self.Q = np.eye(4) * 0.01

        # measurement noise
        self.R = np.eye(2) * 0.25

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z).reshape((2, 1))

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2].flatten()


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