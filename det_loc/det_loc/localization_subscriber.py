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
            Image,  # use CompressedImage or Image
            "/image_raw",
            self.listener_callback,
            10,
        )

        # self.subscription = self.create_subscription(
        #     Image,  # use CompressedImage or Image
        #     "/image_raw",
        #     self.listener_callback,
        #     10,
        # )

        self.camera_params = None
        self.camera_info_received = False
        self.create_subscription(
            CameraInfo, "/camera_info", self.camera_info_callback, 10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.br = CvBridge()
        self.last_line = None
        self.initial_camera_timer = self.create_timer(1.0, self.initial_camera_position)
        self.in_rosbag = True
        self.angular_speed_list = []
        self.detector = apriltag("tagStandard41h12")
        self.tag_size = 0.160
        self.kf = KalmanFilter2D(dt=0.05)  # should probs try different dt vals
        self.target = [2.850, 3.0]  # penalty spot

    def camera_info_callback(self, msg: CameraInfo):
        """Get camera info"""
        if not self.camera_info_received and msg is not None:
            K = msg.k  # 3×3 calibration matrix as a flat list
            fx, fy = K[0], K[4]
            cx, cy = K[2], K[5]

            self.camera_params = [fx, fy, cx, cy]
            self.camera_info_received = True

            self.get_logger().info(
                f"Camera intrinsics received: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
            )

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

        frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")  # for Image
        # np_arr = np.frombuffer(data.data, np.uint8)  # for CompressedImage
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # for CompressedImage

        # cv2.imshow("frame", frame)

        # Load and preprocess grayscale for detection
        processed_frame = preprocess_image(frame)

        # Detect AprilTags
        detections = self.detector.detect(processed_frame)
        # detections["lb-rb-rt-lt"] = detections["lb-rb-rt-lt"] / 2

        # Draw bounding boxes on original color image
        vis_image = draw_detections(frame, detections, scale=2.0)

        cv2.imshow("detected tags", vis_image)

        distance_frame, distances = distance_measure(
            vis_image,
            detections,
            self.camera_params,
            self.tag_size,
            self.get_logger(),
        )

        # cv2.imshow("distances", distance_frame)

        if len(detections) > 1:
            if len(detections) > 2:
                robot_pos = triangulation_3p(detections, distances)
            else:
                robot_pos = triangulation_2p(detections, distances)

            print(f"robot_pos: {robot_pos}")

            self.kf.predict()
            filtered_pos = self.kf.update(robot_pos)

            self.get_logger().info(
                f"Filtered position: x={filtered_pos[0]:.2f}, y={filtered_pos[1]:.2f}"
            )

        # ### curling algorithm ###
        # twist = Twist()

        # ducky_pos = self.target
        # # calculate the angle between the rover and ducky.
        # rotation = np.arctan2(ducky_pos[1] - robot_pos[1], ducky_pos[0] - robot_pos[0])
        # # calculate how much the rover needs to turn and normalize (in radians)
        # z_error = rotation - robot_pos[2]
        # z_error = (z_error + np.pi) % (2 * np.pi) - np.pi

        # # check how far away you are (in m)
        # distance_to_ducky = np.sqrt(
        #     (robot_pos[0] - ducky_pos[0]) ** 2 + (robot_pos[1] - ducky_pos[1]) ** 2
        # )

        # # if you are close, reduce speed to be more precise
        # if distance_to_ducky < 0.5:
        #     speed = 0.1
        #     turning = 0.3
        # else:
        #     speed = 0.3
        #     turning = 0.5

        # if z_error < 0.3:
        #     # drive fowards if in the right direction
        #     twist.linear.x = speed
        #     twist.angular.z = 0.0
        # else:
        #     twist.linear.x = 0
        #     # otherwise turn to the right direction
        #     if z_error > 0.3:
        #         twist.angular.z = turning
        #     else:
        #         twist.angular.z = -turning

        # if distance_to_ducky < 0.10:
        #     # if within 10 cm of the ducky, stop
        #     twist.linear.x = 0.0
        #     twist.angular.z = 0.0

        cv2.waitKey(1)


def preprocess_image(color_image):
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    scale = 2.0
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image


def draw_detections(color_image, detections, scale=2.0):
    """
    Draw bounding boxes and tag IDs on the color image.

    Args:
        color_image: Original RGB image
        detections: List of AprilTag detections
        scale: Scale factor used during preprocessing (to adjust coordinates)
    """
    vis_image = color_image.copy()

    for detection in detections:
        # Get corner coordinates and scale them back to original image size
        corners = detection["lb-rb-rt-lt"]
        corners = [(int(x / scale), int(y / scale)) for x, y in corners]
        # corners = [(int(x), int(y)) for x, y in corners]

        # Draw bounding box
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i + 1) % 4]
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)

        # Get center point and scale back
        center = (
            int(detection["center"][0] / scale),
            int(detection["center"][1] / scale),
        )

        # Draw center point
        cv2.circle(vis_image, center, 5, (0, 0, 255), -1)

        # Draw tag ID
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


def distance_measure(frame, results, cam_params, tag_size, logger):
    """
    Measures the distance to an apriltag.
    Hardcoded with camera parameters from 'pt_camera_link'.
    """

    if len(results) == 0:
        return frame, None

    # 1. Set physical size (removed the hardcoded overwrite so the argument works)
    # Ensure tag_size is passed in meters (e.g., 0.288)
    half = tag_size / 2.0

    # 2. GEOMETRY FIX: Match 3D points to 'lb-rb-rt-lt' (Counter-Clockwise starting Bottom-Left)
    # Previous code had Top-Left first, which causes a twist/error.
    object_points = np.array(
        [
            [-half, -half, 0],  # Bottom-Left
            [half, -half, 0],  # Bottom-Right
            [half, half, 0],  # Top-Right
            [-half, half, 0],  # Top-Left
        ],
        dtype=np.float32,
    )

    # 3. Hardcoded Intrinsics from your 'k' list
    fx = 289.11451
    fy = 289.75319
    cx = 347.23664
    cy = 235.67429

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # 4. Hardcoded Distortion from your 'd' list
    dist_coeffs = np.array(
        [-0.208848, 0.028006, -0.000705, -0.00082, 0.0], dtype=np.float32
    )

    distances = []
    for det in results:
        # Ensure corners are float32
        corners = np.array(det["lb-rb-rt-lt"], dtype=np.float32)

        # Check if corners are properly shaped (4, 2)
        if corners.shape == (4, 2):
            # solvePnP requires contiguous arrays
            corners = np.ascontiguousarray(corners)

            success, rvec, tvec = cv2.solvePnP(
                object_points, corners, camera_matrix, dist_coeffs
            )

            if success:
                # Euclidean distance
                dist = float(np.linalg.norm(tvec))
                distances.append(dist)
            else:
                distances.append(None)
        else:
            distances.append(None)

    # Logging
    for i in range(len(distances)):
        if distances[i] is not None:
            logger.info(f"AprilTag {results[i]['id']} distance: {distances[i]:.3f} m")

    return frame, distances


def get_rotation(ax, ay, bx, by, robot_x, robot_y):
    ### might not be good. Use rvec from pnp? ###

    # compute angles from each point to camera (in radians)
    theta1 = np.arctan2(ay - robot_y, ax - robot_x)
    theta2 = np.arctan2(by - robot_y, bx - robot_y)

    # take the average angle
    theta_camera = (theta1 + theta2) / 2

    # go from camera to robot orientation

    return theta_camera


def triangulation_3p(detections, distances):
    """Performs triangulation to compute the position of the Robot with 3 points."""
    # get intersections between all three points
    [ab_x, ab_y, ab_z] = triangulation_2p(detections[:2], distances[:2])
    [ac_x, ac_y, ac_z] = triangulation_2p(
        [detections[0], detections[2]], [distances[0], distances[2]]
    )
    [bc_x, bc_y, bc_z] = triangulation_2p(detections[1:3], distances[1:3])

    # get mean of intersections and rotations
    x = np.mean(np.array([ab_x, ac_x, bc_x]))
    y = np.mean(np.array([ab_y, ac_y, bc_y]))
    z = np.mean(np.array([ab_z, ac_z, bc_z]))

    return [x, y, z]


def triangulation_2p(detections, distances):
    """Performs triangulation to compute the position of the Robot with 2 points."""
    # print(detections, distances)
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
    [ax, ay] = april_tags[f"tag{a}"]
    [bx, by] = april_tags[f"tag{b}"]

    [a_d, b_d] = distances

    # get euclidean distance between a and b
    d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

    if d > a_d + b_d:
        print("circles dont intersect")

    # get triangle
    z = (a_d**2 - b_d**2 + d**2) / (2 * d)
    h = np.sqrt(a_d**2 - z**2)
    cx = ax + z * (bx - ax) / d
    cy = ay + z * (by - ay) / d

    # get possible robot coordinates
    qx = cx + h * (by - ay) / d
    qy = cy - h * (bx - ax) / d
    px = cx - h * (by - ay) / d
    py = cy + h * (bx - ax) / d

    # remove the extraneous solution that lies outside the field.
    if qx < 0 or qx > 9:  # field measured in m
        robot_pos = [px, py]
    elif qy < 0 or qy > 6:  # field measured in m
        robot_pos = [px, py]
    else:
        robot_pos = [qx, qy]

    robot_pos.append(get_rotation(ax, ay, bx, by, robot_pos[0], robot_pos[1]))

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
        # return self.x

    def update(self, z):
        z = np.array(z).reshape((2, 1))

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2].flatten()  # or return self.x?


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
