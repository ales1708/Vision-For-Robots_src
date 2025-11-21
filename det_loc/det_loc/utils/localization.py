"""
Robot Localization Module

This module handles distance measurement, triangulation, and Kalman filtering
for robot position estimation using AprilTag detections.
"""

import numpy as np
import cv2


# AprilTag positions in the field (in meters)
APRILTAG_POSITIONS = {
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

# Field boundaries (in meters)
FIELD_WIDTH = 9.0
FIELD_HEIGHT = 6.0


class DistanceMeasurement:
    """Handles distance measurement to AprilTags using camera calibration."""

    def __init__(self, camera_params, tag_size=0.288):
        """
        Initialize distance measurement.

        Args:
            camera_params: Dict with 'fx', 'fy', 'cx', 'cy', 'distortion'
            tag_size: Physical size of AprilTag in meters (default: 0.288m)
        """
        self.tag_size = tag_size
        self.half_size = tag_size / 2.0

        # Camera intrinsics
        fx = camera_params.get('fx', 289.11451)
        fy = camera_params.get('fy', 289.75319)
        cx = camera_params.get('cx', 347.23664)
        cy = camera_params.get('cy', 235.67429)

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients
        self.dist_coeffs = np.array(
            camera_params.get('distortion', [-0.208848, 0.028006, -0.000705, -0.00082, 0.0]),
            dtype=np.float32
        )

        # 3D object points (tag corners in tag's coordinate system)
        # Matches 'lb-rb-rt-lt' order (Counter-Clockwise from Bottom-Left)
        self.object_points = np.array([
            [-self.half_size, -self.half_size, 0],  # Bottom-Left
            [self.half_size, -self.half_size, 0],   # Bottom-Right
            [self.half_size, self.half_size, 0],    # Top-Right
            [-self.half_size, self.half_size, 0],   # Top-Left
        ], dtype=np.float32)

    def measure_distances(self, detections, logger=None):
        """
        Measure distances to detected AprilTags.

        Args:
            detections: List of AprilTag detections
            logger: Optional ROS logger for info messages

        Returns:
            List of distances (in meters) or None for failed measurements
        """
        if not detections:
            return []

        distances = []
        for det in detections:
            corners = np.array(det["lb-rb-rt-lt"], dtype=np.float32)

            if corners.shape == (4, 2):
                corners = np.ascontiguousarray(corners)
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points,
                    corners,
                    self.camera_matrix,
                    self.dist_coeffs
                )

                if success:
                    dist = float(np.linalg.norm(tvec))
                    distances.append(dist)

                    if logger:
                        logger.info(f"Tag {det['id']} distance: {dist:.3f} m")
                else:
                    distances.append(None)
            else:
                distances.append(None)

        return distances


class Triangulation:
    """Handles robot position triangulation from multiple AprilTag detections."""

    @staticmethod
    def _get_rotation(ax, ay, bx, by, robot_x, robot_y):
        """
        Estimate robot rotation from two tag positions.

        Args:
            ax, ay: Position of first tag
            bx, by: Position of second tag
            robot_x, robot_y: Robot position

        Returns:
            Estimated rotation angle (radians)
        """
        theta1 = np.arctan2(ay - robot_y, ax - robot_x)
        theta2 = np.arctan2(by - robot_y, bx - robot_x)
        return (theta1 + theta2) / 2

    @staticmethod
    def triangulate_2p(detections, distances, tag_positions=None):
        """
        Compute robot position from 2 AprilTag detections using triangulation.

        Args:
            detections: List of 2 detection dicts with 'id' key
            distances: List of 2 distances (in meters)
            tag_positions: Optional dict of tag positions (uses APRILTAG_POSITIONS if None)

        Returns:
            [x, y, rotation] position or None if invalid
        """
        if len(detections) < 2 or len(distances) < 2:
            return None

        if distances[0] is None or distances[1] is None:
            return None

        positions = tag_positions or APRILTAG_POSITIONS

        # Get tag positions
        tag_a_id = detections[0]["id"]
        tag_b_id = detections[1]["id"]

        tag_a_key = f"tag{tag_a_id}"
        tag_b_key = f"tag{tag_b_id}"

        if tag_a_key not in positions or tag_b_key not in positions:
            print(f"Warning: Unknown tags {tag_a_id} or {tag_b_id}")
            return None

        ax, ay = positions[tag_a_key]
        bx, by = positions[tag_b_key]
        a_d, b_d = distances

        # Distance between tags
        d = np.sqrt((bx - ax)**2 + (by - ay)**2)

        # Check if circles intersect
        if d > a_d + b_d:
            print("Warning: Circles don't intersect - measurement error")
            return None

        # Circle intersection geometry
        z = (a_d**2 - b_d**2 + d**2) / (2 * d)

        # Check if discriminant is valid
        discriminant = a_d**2 - z**2
        if discriminant < 0:
            print("Warning: Invalid triangle geometry")
            return None

        h = np.sqrt(discriminant)
        cx = ax + z * (bx - ax) / d
        cy = ay + z * (by - ay) / d

        # Two possible robot positions
        qx = cx + h * (by - ay) / d
        qy = cy - h * (bx - ax) / d
        px = cx - h * (by - ay) / d
        py = cy + h * (bx - ax) / d

        # Choose solution within field boundaries
        if 0 <= qx <= FIELD_WIDTH and 0 <= qy <= FIELD_HEIGHT:
            robot_pos = [qx, qy]
        else:
            robot_pos = [px, py]

        # Estimate rotation
        rotation = Triangulation._get_rotation(ax, ay, bx, by, robot_pos[0], robot_pos[1])
        robot_pos.append(rotation)

        return robot_pos

    @staticmethod
    def triangulate_3p(detections, distances, tag_positions=None):
        """
        Compute robot position from 3 AprilTag detections (more accurate).

        Args:
            detections: List of 3 detection dicts
            distances: List of 3 distances
            tag_positions: Optional dict of tag positions

        Returns:
            [x, y, rotation] average position or None if invalid
        """
        if len(detections) < 3 or len(distances) < 3:
            return None

        # Get three 2-point triangulations
        ab_pos = Triangulation.triangulate_2p(
            detections[:2], distances[:2], tag_positions
        )
        ac_pos = Triangulation.triangulate_2p(
            [detections[0], detections[2]],
            [distances[0], distances[2]],
            tag_positions
        )
        bc_pos = Triangulation.triangulate_2p(
            detections[1:3], distances[1:3], tag_positions
        )

        # Filter out None results
        valid_positions = [pos for pos in [ab_pos, ac_pos, bc_pos] if pos is not None]

        if not valid_positions:
            return None

        # Average valid positions
        positions_array = np.array(valid_positions)
        x = np.mean(positions_array[:, 0])
        y = np.mean(positions_array[:, 1])
        rotation = np.mean(positions_array[:, 2])

        return [x, y, rotation]


class KalmanFilter2D:
    """Kalman filter for smoothing 2D position estimates."""

    def __init__(self, dt=0.05, process_noise=0.01, measurement_noise=0.25):
        """
        Initialize 2D Kalman filter.

        Args:
            dt: Time step (seconds)
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        # State vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # Observation matrix (measure only x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Covariance matrix
        self.P = np.eye(4)

        # Process noise
        self.Q = np.eye(4) * process_noise

        # Measurement noise
        self.R = np.eye(2) * measurement_noise

    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        Update state with new measurement.

        Args:
            measurement: [x, y] position measurement

        Returns:
            Filtered [x, y] position
        """
        z = np.array(measurement).reshape((2, 1))

        # Innovation
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2].flatten()

    def get_state(self):
        """Get current state [x, y, vx, vy]."""
        return self.x.flatten()

