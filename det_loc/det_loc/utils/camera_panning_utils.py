from sensor_msgs.msg import JointState
import numpy as np

class Pan_Controller:
    def __init__(self, pan_publisher):
        self.pan_position = 0.0
        self.pan_step = 0.05  # Smaller step for slower, more accurate scanning
        self.pan_min = -1.25
        self.pan_max = 1.25
        self.pan_direction = 1
        self.pan_publisher = pan_publisher
        self.pan_max_reached = False
        self.pan_min_reached = False
        self.scanning_complete = False

        # Dynamic tracking parameters
        self.tracking_gain = 0.003  # radians per pixel of error
        self.max_tracking_adjustment = 0.1  # max radians per adjustment

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

    def adjust_pan_to_center(self, center_error_x, image_center_x):
        """Calculate required pan adjustment to center detected tags
        """
        adjustment = -center_error_x * self.tracking_gain
        adjustment = np.clip(adjustment, -self.max_tracking_adjustment, self.max_tracking_adjustment)

        return adjustment

    def apply_tracking_adjustment(self, adjustment):
        """Apply pan adjustment for tracking and publish new position
        """
        new_position = self.pan_position + adjustment

        # Respect pan limits
        if new_position < self.pan_min or new_position > self.pan_max:
            # Clamp to limits
            new_position = np.clip(new_position, self.pan_min, self.pan_max)
            if new_position == self.pan_position:
                # Already at limit, no adjustment possible
                return False

        self.pan_position = new_position
        self.publish_pan_position()
        return True

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

        self.scan_data = []  # List of dicts with {pan_position, num_detections, center_error}
        self.current_scan_accumulator = []  # Accumulate data for current pan position
        self.frames_per_position = 1  # Number of frames to average per position

        # Tracking state management
        self.tracking_enabled = False
        self.center_error_threshold = 60.0  # pixels from center before adjustment

    def initial_scanning(self):
        """Reset and start scanning operation"""
        self.pan_controller.reset_scanning_position()
        self.scan_data = []
        self.current_scan_accumulator = []

        # Reset best view tracking for fresh scan
        self.best_view = None
        self.best_view_score = 0.0
        self.best_view_pan_position = 0.0

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

        print("before avg_pan")
        avg_pan = np.mean([d['pan_position'] for d in self.current_scan_accumulator])
        print("before avg_detections")
        avg_detections = np.mean([d['num_detections'] for d in self.current_scan_accumulator])
        print("before avg_center")
        avg_center_error = np.mean([d['center_error'] for d in self.current_scan_accumulator if d['center_error'] != float('inf')])

        if np.isnan(avg_center_error):
            avg_center_error = float('inf')

        return {
            'pan_position': avg_pan,
            'num_detections': avg_detections,
            'center_error': avg_center_error
        }

    def track_view(self, scan_data_point):
        """Track the best view based on number of detections and center error."""
        num_detections = scan_data_point['num_detections']
        center_error = scan_data_point['center_error']

        if num_detections == 0 or center_error == float('inf'):
            score = 0.0
        else:
            max_center_error = np.sqrt(self.image_center[0]**2 + self.image_center[1]**2)
            normalized_error = min(center_error / max_center_error, 1.0)
            score = num_detections * (1.0 - normalized_error)

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
            center_x = detection['center'][0]
            center_y = detection['center'][1]

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

    def enable_tracking(self):
        """Enable dynamic tracking mode after scanning completes"""
        self.tracking_enabled = True

    def disable_tracking(self):
        """Disable dynamic tracking mode"""
        self.tracking_enabled = False

    def check_and_adjust_tracking(self, detections):
        """Check if tags need re-centering and adjust pan if necessary
        """
        if not self.tracking_enabled or not detections:
            return False, 0.0, 0.0

        avg_center_x = np.mean([det['center'][0] for det in detections])
        error_x = avg_center_x - self.image_center[0]

        if abs(error_x) > self.center_error_threshold:
            adjustment = self.pan_controller.adjust_pan_to_center(error_x, self.image_center[0])
            success = self.pan_controller.apply_tracking_adjustment(adjustment)

            if success:
                return True, error_x, adjustment
            else:
                return False, error_x, 0.0

        return False, error_x, 0.0