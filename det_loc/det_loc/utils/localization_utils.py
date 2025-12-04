import cv2
import numpy as np


def distance_measure(frame, results, camera_matrix, tag_size, logger):
    """
    Measures the distance to AprilTags and returns detailed detection info.
    Uses the Undistorted Camera Matrix (P) and 0 distortion.

    Returns:
        frame: Original frame (unchanged)
        detections_info: List of dicts with {'id': int, 'tvec': np.array, 'distance': float}
        rvecs: List of rotation vectors (kept for backward compatibility)
    """

    if len(results) == 0:
        return frame, [], None

    half = tag_size / 2.0

    # Counter-Clockwise starting Bottom-Left
    object_points = np.array(
        [
            [-half, -half, 0],  # Bottom-Left
            [half, -half, 0],  # Bottom-Right
            [half, half, 0],  # Top-Right
            [-half, half, 0],  # Top-Left
        ],
        dtype=np.float32,
    )

    # Since the image passed in is already undistorted, we assume 0 distortion for solvePnP
    dist_coeffs = np.zeros((4, 1))

    detections_info = []
    rvecs = []

    for det in results:
        # Get corners from detection
        corners = np.array(det["lb-rb-rt-lt"], dtype=np.float32)

        if corners.shape == (4, 2):
            corners = np.ascontiguousarray(corners)

            success, rvec, tvec = cv2.solvePnP(
                object_points, corners, camera_matrix, dist_coeffs
            )

            if success:
                dist = float(np.linalg.norm(tvec))
                detections_info.append(
                    {
                        "id": det["id"],
                        "tvec": tvec.flatten(),  # Store as 1D array [x, y, z]
                        "distance": dist,
                    }
                )
                rvecs.append(rvec)
            else:
                detections_info.append(
                    {"id": det["id"], "tvec": None, "distance": None}
                )
                rvecs.append(None)
        else:
            detections_info.append({"id": det["id"], "tvec": None, "distance": None})
            rvecs.append(None)

    return frame, detections_info, rvecs


def triangulation_3p(detections_info):
    """
    Performs triangulation to compute the position of the Robot with 3 points.

    Args:
        detections_info: List of dicts with {'id': int, 'tvec': np.array, 'distance': float}

    Returns:
        [x, y]: Robot position
        rotation: Robot yaw angle in radians (normalized to [-π, π])
    """
    # Get intersections between all three points
    ab_pos, ab_rot = triangulation_2p([detections_info[0], detections_info[1]])
    ac_pos, ac_rot = triangulation_2p([detections_info[0], detections_info[2]])
    bc_pos, bc_rot = triangulation_2p([detections_info[1], detections_info[2]])

    # Get mean of intersections for position
    x = np.mean(np.array([ab_pos[0], ac_pos[0], bc_pos[0]]))
    y = np.mean(np.array([ab_pos[1], ac_pos[1], bc_pos[1]]))

    # Use circular mean for rotation angles
    rotation = circular_mean([ab_rot, ac_rot, bc_rot])

    return [x, y], rotation


def circular_mean(angles):
    """
    Compute the circular mean of angles to properly average rotations.

    Args:
        angles: List of angles in radians

    Returns:
        Mean angle in radians, normalized to [-π, π]
    """
    sin_sum = np.sum([np.sin(a) for a in angles])
    cos_sum = np.sum([np.cos(a) for a in angles])
    mean_angle = np.arctan2(sin_sum, cos_sum)
    return mean_angle


def triangulation_2p(detections_info):
    """
    Performs triangulation to compute the position and orientation of the Robot with 2 points.
    Uses Vector Alignment method for rotation calculation.

    Args:
        detections_info: List of 2 dicts with {'id': int, 'tvec': np.array, 'distance': float}

    Returns:
        robot_pos: [x, y] position
        robot_yaw: Rotation angle in radians (normalized to [-π, π])
    """
    # Extract tag IDs and distances
    tag_a = detections_info[0]
    tag_b = detections_info[1]

    a_d = tag_a["distance"]
    b_d = tag_b["distance"]

    if a_d is None or b_d is None:
        return [0.0, 0.0], 0.0

    # Get tag positions in world coordinates
    ax, ay, bx, by = get_tag_positions_from_ids(tag_a["id"], tag_b["id"])

    # Get euclidean distance between a and b
    d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

    # Triangle inequality check (with slight tolerance)
    if d > (a_d + b_d) * 1.05:
        print("Warning: Circles don't intersect properly")

    # Get triangle geometry for position
    try:
        z = (a_d**2 - b_d**2 + d**2) / (2 * d)
        h = np.sqrt(max(0, a_d**2 - z**2))  # prevent sqrt of negative
        cx = ax + z * (bx - ax) / d
        cy = ay + z * (by - ay) / d

        # Get possible robot coordinates
        qx = cx + h * (by - ay) / d
        qy = cy - h * (bx - ax) / d
        px = cx - h * (by - ay) / d
        py = cy + h * (bx - ax) / d

        # Remove the extraneous solution that lies outside the field
        if qx < 0 or qx > 9 or qy < 0 or qy > 6:
            robot_pos = [px, py]
        else:
            robot_pos = [qx, qy]

    except Exception as e:
        print(f"Triangulation error: {e}")
        return [0.0, 0.0], 0.0

    # ========== VECTOR ALIGNMENT METHOD FOR ROTATION ==========

    # 1. Calculate World Vector angle (from Tag A to Tag B in map coordinates)
    world_angle = np.arctan2(by - ay, bx - ax)

    # 2. Calculate Camera Vector angle (from Tag A to Tag B in camera frame)
    tvec_a = tag_a["tvec"]
    tvec_b = tag_b["tvec"]

    if tvec_a is None or tvec_b is None:
        return robot_pos, 0.0

    # OpenCV convention: Z is forward, X is right, Y is down
    # Calculate vector from A to B in camera frame
    delta_x_cv = tvec_b[0] - tvec_a[0]  # Right difference
    delta_z_cv = tvec_b[2] - tvec_a[2]  # Forward difference

    # For ROS 2D nav: Forward=X, Left=Y
    # So we need: atan2(-ΔX_cv, ΔZ_cv)
    # Negative X because in OpenCV +X is right, but we need left for positive rotation
    camera_angle = np.arctan2(-delta_x_cv, delta_z_cv)

    # 3. Calculate Robot Yaw
    robot_yaw = world_angle - camera_angle

    # 4. Normalize to [-π, π]
    robot_yaw = normalize_angle(robot_yaw)

    return robot_pos, robot_yaw


def normalize_angle(angle):
    """
    Normalize angle to [-π, π] range.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in radians
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_tag_positions_from_ids(id_a, id_b):
    """
    Retrieves the location of detected tags by their IDs.

    Args:
        id_a: First tag ID
        id_b: Second tag ID

    Returns:
        ax, ay, bx, by: x and y coordinates of both tags
    """
    april_tags = {
        1: [9.0, 3.0],
        2: [7.350, 0.0],
        3: [7.350, 6.0],
        4: [4.5, 0.0],
        5: [4.5, 6.0],
        6: [2.644, -0.28],
        7: [2.644, 6.78],
        8: [0, 0.144],
        9: [0.0, 3.0],
        10: [0.0, 5.866],
    }

    # Verify tags exist in dict
    if id_a not in april_tags or id_b not in april_tags:
        print(f"Tag ID {id_a} or {id_b} not found in map")
        return 0.0, 0.0, 0.0, 0.0

    ax, ay = april_tags[id_a]
    bx, by = april_tags[id_b]

    return ax, ay, bx, by


def get_tag_positions(detections):
    """
    DEPRECATED: Use get_tag_positions_from_ids instead.
    Kept for backward compatibility.
    """
    a = detections[0]["id"]
    b = detections[1]["id"]
    return get_tag_positions_from_ids(a, b)


# ========== KALMAN FILTER (UNCHANGED) ==========


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
