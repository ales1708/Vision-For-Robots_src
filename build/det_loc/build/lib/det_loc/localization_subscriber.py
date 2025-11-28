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
            Image,
            "/manual_image_rect",
            self.listener_callback,
            10,
        )

        self.camera_params = None
        self.camera_info_received = False
        self.create_subscription(
            CameraInfo, "/camera_info", self.camera_info_callback, 10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.br = CvBridge()
        self.initial_camera_timer = self.create_timer(1.0, self.initial_camera_position)
        self.angular_speed_list = []
        self.detector = apriltag("tagStandard41h12")
        self.tag_size = 0.160
        self.kf = KalmanFilter2D(dt=0.05)
        self.target = [2.850, 3.0]  # penalty spot

    def camera_info_callback(self, msg: CameraInfo):
        """Get camera info"""
        if not self.camera_info_received and msg is not None:
            K = msg.k
            fx, fy = K[0], K[4]
            cx, cy = K[2], K[5]
            self.camera_params = [fx, fy, cx, cy]
            self.camera_info_received = True
            self.get_logger().info(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}")

    def initial_camera_position(self):
        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        msg.position = [0.0, 0.0]
        self.joint_pub.publish(msg)
        self.get_logger().info("Camera turned to preset position.")
        self.destroy_timer(self.initial_camera_timer)

    def listener_callback(self, data):
        frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        processed_frame = preprocess_image(frame)

        # Detect AprilTags
        detections = self.detector.detect(processed_frame)

        # Draw bounding boxes
        vis_image = draw_detections(frame, detections, scale=2.0)
        cv2.imshow("detected tags", vis_image)

        # --- UPDATED FUNCTION CALL ---
        # Now returns tag_poses (list of dicts) instead of just distances
        distance_frame, tag_poses = distance_measure(
            vis_image,
            detections,
            self.camera_params,
            self.tag_size,
            self.get_logger(),
        )

        robot_pos = None

        if len(tag_poses) > 1:
            if len(tag_poses) > 2:
                # Pass tag_poses directly
                robot_pos = triangulation_3p(tag_poses)
            else:
                robot_pos = triangulation_2p(tag_poses)

        if robot_pos is not None:
            print(f"robot_pos: {robot_pos}")
            self.kf.predict()
            filtered_pos = self.kf.update(robot_pos)

            self.get_logger().info(
                f"Filtered: x={filtered_pos[0]:.2f}, y={filtered_pos[1]:.2f}"
            )

            ### curling algorithm ###
            twist = Twist()
            ducky_pos = self.target

            rotation = np.arctan2(ducky_pos[1] - robot_pos[1], ducky_pos[0] - robot_pos[0])
            
            # NOTE: robot_pos[2] comes from triangulation. 
            # Ensure triangulation returns theta in index 2.
            z_error = rotation - robot_pos[2]
            z_error = (z_error + np.pi) % (2 * np.pi) - np.pi

            distance_to_ducky = np.sqrt(
                (robot_pos[0] - ducky_pos[0]) ** 2 + (robot_pos[1] - ducky_pos[1]) ** 2
            )

            if distance_to_ducky < 0.5:
                speed = 0.1
                turning = 0.3
            else:
                speed = 0.3
                turning = 0.5

            if abs(z_error) < 0.3:
                twist.linear.x = speed
                twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                if z_error > 0.0:
                    twist.angular.z = turning
                else:
                    twist.angular.z = -turning

            if distance_to_ducky < 0.10:
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            # self.cmd_vel_pub.publish(twist) # Uncomment to drive
            cv2.waitKey(1)
        else:
            cv2.waitKey(1)


def preprocess_image(color_image):
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    scale = 2.0
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image


def draw_detections(color_image, detections, scale=2.0):
    vis_image = color_image.copy()
    for detection in detections:
        corners = detection["lb-rb-rt-lt"]
        corners = [(int(x / scale), int(y / scale)) for x, y in corners]

        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i + 1) % 4]
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)

        center = (
            int(detection["center"][0] / scale),
            int(detection["center"][1] / scale),
        )
        cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
        cv2.putText(vis_image,f"ID: {detection['id']}",(center[0] - 20, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0),2)
    return vis_image


# --- UPDATED DISTANCE MEASURE ---
def distance_measure(frame, results, cam_params, tag_size, logger):
    """
    Returns frame and list of dicts: {'id', 'tvec', 'distance'}
    """
    if len(results) == 0:
        return frame, []

    half = tag_size / 2.0
    object_points = np.array(
        [
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
        ],
        dtype=np.float32,
    )

    fx = 289.11451
    fy = 289.75319
    cx = 347.23664
    cy = 235.67429
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.array(
        [-0.208848, 0.028006, -0.000705, -0.00082, 0.0], dtype=np.float32
    )

    tag_poses = []

    for det in results:
        corners = np.array(det["lb-rb-rt-lt"], dtype=np.float32)

        if corners.shape == (4, 2):
            corners = np.ascontiguousarray(corners)
            success, rvec, tvec = cv2.solvePnP(
                object_points, corners, camera_matrix, dist_coeffs
            )

            if success:
                dist = float(np.linalg.norm(tvec))
                # Store everything we need in a clean dictionary
                tag_data = {
                    "id": det['id'],
                    "tvec": tvec, 
                    "rvec": rvec, # Added rvec in case you need it later
                    "distance": dist
                }
                tag_poses.append(tag_data)

    # Logging
    for data in tag_poses:
        logger.info(f"ID: {data['id']} | Dist: {data['distance']:.3f} m")

    return frame, tag_poses


def get_rotation(ax, ay, bx, by, robot_x, robot_y):
    # compute angles from each point to camera (in radians)
    theta1 = np.arctan2(ay - robot_y, ax - robot_x)
    theta2 = np.arctan2(by - robot_y, bx - robot_x) # Typo fix: bx - robot_x (was robot_y)
    
    # take the average angle
    theta_camera = (theta1 + theta2) / 2
    return theta_camera


# --- UPDATED TRIANGULATION FUNCTIONS ---
def triangulation_3p(tag_poses):
    """
    Performs triangulation with 3 points. 
    Accepts list of dicts from distance_measure.
    """
    # We pass slices of the list of dicts directly
    [ab_x, ab_y, ab_z] = triangulation_2p(tag_poses[:2])
    
    # Create temp lists for the other pairs
    pair_ac = [tag_poses[0], tag_poses[2]]
    pair_bc = [tag_poses[1], tag_poses[2]]
    
    [ac_x, ac_y, ac_z] = triangulation_2p(pair_ac)
    [bc_x, bc_y, bc_z] = triangulation_2p(pair_bc)

    x = np.mean(np.array([ab_x, ac_x, bc_x]))
    y = np.mean(np.array([ab_y, ac_y, bc_y]))
    z = np.mean(np.array([ab_z, ac_z, bc_z]))

    # Note: This returns x,y,z (height). 
    # If your logic expects [x,y,theta], you might need to adjust this return.
    return [x, y, z]


def triangulation_2p(tag_poses):
    """
    Performs triangulation with 2 points.
    Accepts list of dicts: [{'id': 1, 'distance': 1.0, ...}, ...]
    """
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

    # Extract data from the dictionary structure
    tag_a = tag_poses[0]
    tag_b = tag_poses[1]

    id_a = tag_a["id"]
    id_b = tag_b["id"]
    
    # If you want to use tvec later for rotation, you can access it here:
    # tvec_a = tag_a["tvec"]

    [ax, ay] = april_tags[f"tag{id_a}"]
    [bx, by] = april_tags[f"tag{id_b}"]

    a_d = tag_a["distance"]
    b_d = tag_b["distance"]

    d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

    if d > a_d + b_d:
        print("circles dont intersect")

    z = (a_d**2 - b_d**2 + d**2) / (2 * d)
    
    # Safety check for sqrt
    term = a_d**2 - z**2
    if term < 0:
        h = 0
    else:
        h = np.sqrt(term)
        
    cx = ax + z * (bx - ax) / d
    cy = ay + z * (by - ay) / d

    qx = cx + h * (by - ay) / d
    qy = cy - h * (bx - ax) / d
    px = cx - h * (by - ay) / d
    py = cy + h * (bx - ax) / d

    if qx < 0 or qx > 9:
        robot_pos = [px, py]
    elif qy < 0 or qy > 6:
        robot_pos = [px, py]
    else:
        robot_pos = [qx, qy]

    # Calculate rotation
    theta = get_rotation(ax, ay, bx, by, robot_pos[0], robot_pos[1])
    robot_pos.append(theta)

    return robot_pos


class KalmanFilter2D:
    def __init__(self, dt=0.05):
        self.x = np.zeros((4, 1))
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(4)
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.25

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Only take x,y from the input (ignore theta if present)
        z = np.array(z[:2]).reshape((2, 1))
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