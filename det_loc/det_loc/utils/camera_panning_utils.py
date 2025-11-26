from sensor_msgs.msg import JointState
import numpy as np

class Pan_Controller:
    def __init__(self, pan_publisher):
        self.pan_position = 0.0
        self.pan_step = 0.1
        self.pan_min = -1.5
        self.pan_max = 1.5
        self.pan_direction = 1
        self.pan_publisher = pan_publisher
        self.pan_max_reached = False
        self.pan_min_reached = False
        self.scanning_complete = False

    def set_pan_position(self, pan_position):
        self.pan_position = pan_position

    def get_pan_position(self):
        return self.pan_position

    def scanning_step(self):
        self.pan_position += self.pan_step * self.pan_direction
        if self.pan_position >= self.pan_max or self.pan_position <= self.pan_min:
            self.pan_direction = -self.pan_direction
            self.pan_position = np.clip(self.pan_position, self.pan_min, self.pan_max)
            if self.pan_position >= self.pan_max:
                self.pan_max_reached = True
            if self.pan_position <= self.pan_min:
                self.pan_min_reached = True

        if self.pan_max_reached and self.pan_min_reached:
            self.pan_direction = 0
            self.scanning_complete = True

        self.publish_pan_position()

    def reset_scanning_position(self):
        self.pan_position = 0.0
        self.pan_direction = 1
        self.pan_max_reached = False
        self.pan_min_reached = False
        self.scanning_complete = False

        self.publish_pan_position()

    def publish_pan_position(self):
        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        msg.position = [self.pan_position, 0.0]
        self.pan_publisher.publish(msg)

class ViewTracker:
    def __init__(self, pan_publisher, image_center=(320, 240)):
        self.best_view = None
        self.best_view_score = 0.0
        self.best_view_pan_position = 0.0
        self.pan_controller = Pan_Controller(pan_publisher)
        self.image_center = image_center

        # Scanning data storage
        self.scan_data = []  # List of dicts with {pan_position, num_detections, center_error}
        self.current_scan_accumulator = []  # Accumulate data for current pan position
        self.frames_per_position = 5  # Number of frames to average per position

    def initial_scanning(self):
        """Reset and start scanning operation"""
        self.pan_controller.reset_scanning_position()
        self.scan_data = []
        self.current_scan_accumulator = []

    def is_scanning_complete(self):
        """Check if scanning operation is complete"""
        return self.pan_controller.scanning_complete

    def update_scan_data(self, detections, pan_position):
        """Update scan data for current pan position"""
        num_detections = len(detections)
        center_error = self.calculate_center_error(detections) if num_detections > 0 else float('inf')

        # Accumulate data for current position
        self.current_scan_accumulator.append({
            'pan_position': pan_position,
            'num_detections': num_detections,
            'center_error': center_error
        })

        # Average over multiple frames at the same position (approximately)
        if len(self.current_scan_accumulator) >= self.frames_per_position:
            avg_data = self._average_scan_data()
            self.scan_data.append(avg_data)
            self.current_scan_accumulator = []

            # Update best view
            self.track_view(avg_data)

    def _average_scan_data(self):
        """Average the accumulated scan data for current position"""
        if not self.current_scan_accumulator:
            return None

        avg_pan = np.mean([d['pan_position'] for d in self.current_scan_accumulator])
        avg_detections = np.mean([d['num_detections'] for d in self.current_scan_accumulator])
        avg_center_error = np.mean([d['center_error'] for d in self.current_scan_accumulator if d['center_error'] != float('inf')])

        if np.isnan(avg_center_error):
            avg_center_error = float('inf')

        return {
            'pan_position': avg_pan,
            'num_detections': avg_detections,
            'center_error': avg_center_error
        }

    def track_view(self, scan_data_point):
        """Track the best view based on number of detections and center error"""
        if scan_data_point is None:
            return

        num_detections = scan_data_point['num_detections']
        center_error = scan_data_point['center_error']

        # Calculate score: prioritize number of detections, then minimize center error
        # Higher score is better
        if num_detections > 0 and center_error != float('inf'):
            score = num_detections * 100 - center_error
        else:
            score = 0.0

        if score > self.best_view_score:
            self.best_view_score = score
            self.best_view = scan_data_point
            self.best_view_pan_position = scan_data_point['pan_position']

    def calculate_center_error(self, detections):
        """Calculate average distance of detections from image center"""
        if not detections:
            return float('inf')

        total_error = 0.0
        for detection in detections:
            # Get center of detection
            center_x = detection['center'][0]
            center_y = detection['center'][1]

            # Calculate Euclidean distance from image center
            error = np.sqrt(
                (center_x - self.image_center[0])**2 +
                (center_y - self.image_center[1])**2
            )
            total_error += error

        return total_error / len(detections)

    def get_best_view(self):
        """Return the best view found during scanning"""
        return self.best_view

    def move_to_best_view(self):
        """Move camera to the best view position"""
        if self.best_view is not None:
            self.pan_controller.set_pan_position(self.best_view_pan_position)
            self.pan_controller.publish_pan_position()
            return True
        return False