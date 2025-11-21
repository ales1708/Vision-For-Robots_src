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
    
    def __init__(self, camera_params, tag_size=0.160):
        """
        Initialize distance measurement.
        
        Args:
            camera_params: Dict with camera calibration parameters
                - 'K': Original camera matrix (3x3)
                - 'D': Distortion coefficients (5 values)
                - 'P': Projection/rectified camera matrix (3x3)
                - Or legacy format: 'fx', 'fy', 'cx', 'cy', 'distortion'
            tag_size: Physical size of AprilTag in meters (default: 0.160m)
        """
        self.tag_size = tag_size
        self.half_size = tag_size / 2.0
        
        # Check if using new calibration format
        if 'K' in camera_params and 'P' in camera_params:
            # New format: undistorted/rectified camera
            self.camera_matrix = camera_params['P']
            self.dist_coeffs = np.zeros((4, 1))  # Already undistorted
            self.use_undistorted = True
        else:
            # Legacy format: original camera with distortion
            fx = camera_params.get('fx', 289.11451)
            fy = camera_params.get('fy', 289.75319)
            cx = camera_params.get('cx', 347.23664)
            cy = camera_params.get('cy', 235.67429)
            
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            self.dist_coeffs = np.array(
                camera_params.get('distortion', [-0.208848, 0.028006, -0.000705, -0.00082, 0.0]),
                dtype=np.float32
            )
            self.use_undistorted = False
        
        # 3D object points (tag corners in tag's coordinate system)
        # Matches 'lb-rb-rt-lt' order (Counter-Clockwise from Bottom-Left)
        self.object_points = np.array([
            [-self.half_size, -self.half_size, 0],  # Bottom-Left
            [self.half_size, -self.half_size, 0],   # Bottom-Right
            [self.half_size, self.half_size, 0],    # Top-Right
            [-self.half_size, self.half_size, 0],   # Top-Left
        ], dtype=np.float32)
    
    def measure_distances(self, detections, scale=1.0, logger=None):
        """
        Measure distances to detected AprilTags.
        
        Args:
            detections: List of AprilTag detections
            scale: Scale factor if detections were done on scaled image
            logger: Optional ROS logger for info messages
            
        Returns:
            List of distances (in meters) or None for failed measurements
        """
        if not detections:
            return []
        
        distances = []
        for det in detections:
            # Get corners from detection
            corners_raw = np.array(det["lb-rb-rt-lt"], dtype=np.float32)
            
            # Scale corners back to original resolution if needed
            corners = corners_raw / scale if scale != 1.0 else corners_raw
            
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
    def triangulate_2p(detections, distances, tag_positions=None):
        """
        Compute robot position from 2 AprilTag detections using triangulation.
        
        Args:
            detections: List of 2 detection dicts with 'id' key
            distances: List of 2 distances (in meters)
            tag_positions: Optional dict of tag positions (uses APRILTAG_POSITIONS if None)
            
        Returns:
            [x, y] position or [0.0, 0.0] if invalid
        """
        if len(detections) < 2 or len(distances) < 2:
            return [0.0, 0.0]
        
        if distances[0] is None or distances[1] is None:
            return [0.0, 0.0]
        
        positions = tag_positions or APRILTAG_POSITIONS
        
        # Get tag positions
        tag_a_id = detections[0]["id"]
        tag_b_id = detections[1]["id"]
        
        tag_a_key = f"tag{tag_a_id}"
        tag_b_key = f"tag{tag_b_id}"
        
        # Verify tags exist in map
        if tag_a_key not in positions or tag_b_key not in positions:
            print(f"Tag ID {tag_a_id} or {tag_b_id} not found in map")
            return [0.0, 0.0]
        
        ax, ay = positions[tag_a_key]
        bx, by = positions[tag_b_key]
        a_d, b_d = distances
        
        # Distance between tags
        d = np.sqrt((bx - ax)**2 + (by - ay)**2)
        
        # Triangle inequality check (with slight tolerance)
        if d > (a_d + b_d) * 1.05:
            print("Circles don't intersect")
        
        # Circle intersection geometry
        try:
            z = (a_d**2 - b_d**2 + d**2) / (2 * d)
            h = np.sqrt(max(0, a_d**2 - z**2))  # Prevent sqrt of negative
            cx = ax + z * (bx - ax) / d
            cy = ay + z * (by - ay) / d
            
            # Two possible robot positions
            qx = cx + h * (by - ay) / d
            qy = cy - h * (bx - ax) / d
            px = cx - h * (by - ay) / d
            py = cy + h * (bx - ax) / d
            
            # Remove the extraneous solution that lies outside the field
            if qx < 0 or qx > FIELD_WIDTH or qy < 0 or qy > FIELD_HEIGHT:
                robot_pos = [px, py]
            else:
                robot_pos = [qx, qy]
                
        except Exception as e:
            print(f"Triangulation error: {e}")
            return [0.0, 0.0]
        
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
            [x, y] average position or [0.0, 0.0] if invalid
        """
        if len(detections) < 3 or len(distances) < 3:
            return [0.0, 0.0]
        
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
        
        # Get mean of intersections
        x = np.mean(np.array([ab_pos[0], ac_pos[0], bc_pos[0]]))
        y = np.mean(np.array([ab_pos[1], ac_pos[1], bc_pos[1]]))
        
        return [x, y]


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

