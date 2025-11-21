"""
Camera Pan Controller for AprilTag Detection

This module handles camera panning logic to find and center AprilTags.
It can be used for initial positioning or dynamic tracking while driving.
"""

import numpy as np
from sensor_msgs.msg import JointState


class CameraController:
    """Manages camera pan/tilt positioning and tag centering logic."""

    def __init__(self, joint_publisher, logger, config=None):
        """
        Initialize camera controller.

        Args:
            joint_publisher: ROS publisher for joint states
            logger: ROS logger instance
            config: Optional dict with configuration parameters
        """
        self.joint_pub = joint_publisher
        self.logger = logger

        # Configuration
        cfg = config or {}
        self.pan_min = cfg.get("pan_min", -1.5)
        self.pan_max = cfg.get("pan_max", 1.5)
        self.pan_step = cfg.get("pan_step", 0.15)
        self.center_threshold_px = cfg.get("center_threshold_px", 60.0)
        self.center_correction_gain = cfg.get("center_correction_gain", 1.2)

        # State
        self.pan_position = 0.0
        self.pan_direction = 1
        self.reached_limits = {"min": False, "max": False}

    def publish_pan_tilt(self, pan, tilt=0.0):
        """Publish camera pan/tilt joint positions."""
        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        msg.position = [float(pan), float(tilt)]
        self.joint_pub.publish(msg)

    def set_pan(self, pan):
        """Set pan position directly."""
        self.pan_position = np.clip(pan, self.pan_min, self.pan_max)
        self.publish_pan_tilt(self.pan_position)

    def center_camera(self):
        """Reset camera to center position."""
        self.pan_position = 0.0
        self.publish_pan_tilt(0.0)

    def calculate_center_error(self, detections, frame_width):
        """
        Calculate horizontal center error of detected tags in pixels.

        Args:
            detections: List of detection dicts with 'center' key
            frame_width: Width of frame in pixels

        Returns:
            Center error in pixels (positive = tags right of center)
            or None if no detections
        """
        if not detections or not frame_width:
            return None
        mean_x = sum(d["center"][0] for d in detections) / len(detections)
        return mean_x - (frame_width / 2.0)

    def should_adjust_to_center(self, detections, frame_width):
        """
        Check if pan should be adjusted to center detected tags.

        Args:
            detections: List of detection dicts
            frame_width: Width of frame in pixels

        Returns:
            True if adjustment needed
        """
        center_error = self.calculate_center_error(detections, frame_width)
        return center_error is not None and abs(center_error) > self.center_threshold_px

    def adjust_pan_to_center(self, detections, frame_width):
        """
        Adjust pan position to center detected tags.

        Args:
            detections: List of detection dicts
            frame_width: Width of frame in pixels
        """
        center_error = self.calculate_center_error(detections, frame_width)
        if center_error is None:
            return

        normalized_error = np.clip(center_error / (frame_width / 2.0), -1.0, 1.0)
        correction = normalized_error * self.pan_step * self.center_correction_gain
        self.pan_position = np.clip(
            self.pan_position + correction,
            self.pan_min,
            self.pan_max
        )
        self.publish_pan_tilt(self.pan_position)

    def sweep_step(self):
        """
        Execute one step of the pan sweep.

        Returns:
            True if sweep is complete (both limits reached)
        """
        self.pan_position += self.pan_direction * self.pan_step

        # Check and handle boundaries
        if self.pan_position >= self.pan_max:
            self.pan_position = self.pan_max
            self.reached_limits["max"] = True
            self.pan_direction = -1
        elif self.pan_position <= self.pan_min:
            self.pan_position = self.pan_min
            self.reached_limits["min"] = True
            self.pan_direction = 1

        self.publish_pan_tilt(self.pan_position)

        return self.reached_limits["min"] and self.reached_limits["max"]

    def reset_sweep(self):
        """Reset sweep state to start a new scan."""
        self.pan_position = 0.0
        self.pan_direction = 1
        self.reached_limits = {"min": False, "max": False}
        self.center_camera()


class ViewTracker:
    """Tracks and scores different camera views to find optimal tag positioning."""

    def __init__(self, logger, config=None):
        """
        Initialize view tracker.

        Args:
            logger: ROS logger instance
            config: Optional dict with configuration parameters
        """
        self.logger = logger

        # Configuration
        cfg = config or {}
        self.center_threshold_px = cfg.get("center_threshold_px", 60.0)
        self.min_tags_for_lock = cfg.get("min_tags_for_lock", 2)
        self.preferred_tags = cfg.get("preferred_tags", 2)

        # State
        self.best_view = None

    def score_view(self, detections, center_error):
        """
        Score a view based on number of tags and centering.

        Args:
            detections: List of detection dicts
            center_error: Center error in pixels

        Returns:
            Score (higher is better) or None
        """
        if not detections or center_error is None:
            return None
        return len(detections) * 1000 - abs(center_error)

    def update_best_view(self, detections, pan_position, center_error):
        """
        Update best view if current view scores higher.

        Args:
            detections: List of detection dicts
            pan_position: Current pan position in radians
            center_error: Center error in pixels

        Returns:
            True if this is a new best view
        """
        if not detections:
            return False

        score = self.score_view(detections, center_error)
        if score is None:
            return False

        if self.best_view is None or score > self.best_view["score"]:
            num_tags = len(detections)
            prev_best = self.best_view

            self.best_view = {
                "pan": pan_position,
                "num_tags": num_tags,
                "center_error": center_error,
                "score": score,
            }

            # Log significant improvements
            if (prev_best is None or
                num_tags > prev_best["num_tags"] or
                abs(center_error) + 10 < abs(prev_best["center_error"])):
                self.logger.info(
                    f"New best: pan={pan_position:.2f}, "
                    f"tags={num_tags}, error={center_error:.1f}px"
                )

            return True

        return False

    def is_view_ideal(self):
        """
        Check if current best view meets ideal criteria.

        Returns:
            True if view has enough centered tags
        """
        if not self.best_view:
            return False
        return (self.best_view["num_tags"] >= self.preferred_tags and
                abs(self.best_view["center_error"]) <= self.center_threshold_px)

    def has_minimum_tags(self):
        """Check if best view has minimum required tags."""
        if not self.best_view:
            return False
        return self.best_view["num_tags"] >= self.min_tags_for_lock

    def get_best_pan(self):
        """Get pan position of best view, or None if no view found."""
        return self.best_view["pan"] if self.best_view else None

    def reset(self):
        """Reset tracking state."""
        self.best_view = None

