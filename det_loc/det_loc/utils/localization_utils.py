import cv2
import numpy as np


def distance_measure(frame, results, camera_matrix, tag_size, logger):
    """
    Measures the distance to an apriltag.
    Uses the Undistorted Camera Matrix (P) and 0 distortion.
    """

    if len(results) == 0:
        return frame, None, None

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

    distances = []
    for det in results:
        # Get corners from detection (which was done on scaled image)
        corners = np.array(det["lb-rb-rt-lt"], dtype=np.float32)

        # CRITICAL: Scale corners back to the original resolution
        # because our camera_matrix corresponds to the original 640x480 resolution.
        # corners = corners_raw / scale

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

    return frame, distances, rvec


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
    # april_tags = {
    #     "tag1": [9.0, 3.0],
    #     "tag2": [7.350, 0.0],
    #     "tag3": [7.350, 6.0],
    #     "tag4": [4.5, 0.0],
    #     "tag5": [4.5, 6.0],
    #     "tag6": [2.644, -0.28],
    #     "tag7": [2.644, 6.78],
    #     "tag8": [0, 0.144],
    #     "tag9": [0.0, 3.0],
    #     "tag10": [0.0, 5.866],
    # }

    # # get detected tags
    # a = detections[0]["id"]
    # b = detections[1]["id"]

    # # Verify tags exist in dict
    # if f"tag{a}" not in april_tags or f"tag{b}" not in april_tags:
    #     print(f"Tag ID {a} or {b} not found in map")
    #     return [0.0, 0.0]

    # [ax, ay] = april_tags[f"tag{a}"]
    # [bx, by] = april_tags[f"tag{b}"]
    ax, ay, bx, by = get_tag_positions(detections)

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
        h = np.sqrt(max(0, a_d**2 - z**2))  # prevent sqrt of negative
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


def get_tag_positions(detections):
    """
    Retrieves the location of detected tags

    Args:
        detections: 2 detected tags

    Returns:
        x and y coordinates of both tags
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

    # get detected tags
    a = detections[0]["id"]
    b = detections[1]["id"]

    # Verify tags exist in dict
    if f"tag{a}" not in april_tags or f"tag{b}" not in april_tags:
        print(f"Tag ID {a} or {b} not found in map")
        return [0.0, 0.0]

    [ax, ay] = april_tags[f"tag{a}"]
    [bx, by] = april_tags[f"tag{b}"]
    return ax, ay, bx, by


def get_tag_rotation(detection):
    april_tags_rotations = {
        # "tag1": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # 180
        # "tag2": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # 90
        # "tag3": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # 270
        # "tag4": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # 90
        # "tag5": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # 270
        # "tag6": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # 90
        # "tag7": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # 270
        # "tag8": np.eye(3),  # 0
        # "tag9": np.eye(3),  # 0
        # "tag10": np.eye(3),  # 0
        "tag1": np.pi,  # 180
        "tag2": 0.5 * np.pi,  # 90
        "tag3": 1.5 * np.pi,  # 270
        "tag4": 0.5 * np.pi,  # 90
        "tag5": 1.5 * np.pi,  # 270
        "tag6": 0.5 * np.pi,  # 90
        "tag7": 1.5 * np.pi,  # 270
        "tag8": 0,  # 0
        "tag9": 0,  # 0
        "tag10": 0,  # 0
    }
    tag = detection["id"]
    rotation = april_tags_rotations[f"tag{tag}"]

    return np.array(rotation), tag


def get_rotation_rvec(rvec, detections):
    """
    Estimate robot rotation from rvec.

    Args:
        rvec: rvec vector obtained from pnp
        detections: detected tags

    Returns:
        Estimated rotation angle (degrees)
    """

    R_cam_to_tag, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R_cam_to_tag[1, 0], R_cam_to_tag[0, 0])
    tag_global_yaw, tag = get_tag_rotation(detections[-1])  # use last detection
    robot_yaw = tag_global_yaw + np.pi + yaw
    # robot_yaw = (robot_yaw + np.pi) % (2 * np.pi) - np.pi # normalize to [-pi, pi]
    robot_yaw = robot_yaw % (2 * np.pi)  # normalize to [0, 2pi]

    robot_yaw_degrees = robot_yaw * 180 / np.pi

    return robot_yaw, robot_yaw_degrees


def get_rotation_from_tags(detections, robot_pos):
    """
    Estimate robot rotation from two tag positions.

    Args:
        detections: detected tags
        robot_pos: x and y coordinates of the robot

    Returns:
        Estimated rotation angle (radians)
    """
    ax, ay, bx, by = get_tag_positions(detections)
    [robot_x, robot_y] = robot_pos

    theta1 = np.arctan2(ay - robot_y, ax - robot_x)
    theta2 = np.arctan2(by - robot_y, bx - robot_x)
    return np.degrees((theta1 + theta2) / 2)


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
