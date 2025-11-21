"""
Detection and Localization Utilities

This package contains utility modules for AprilTag detection, camera control,
and robot localization.
"""

from .camera_controller import CameraController, ViewTracker
from .tag_detector import TagDetector, detect_markers, draw_tag_detections
from .localization import (
    DistanceMeasurement,
    Triangulation,
    KalmanFilter2D,
    DynamicTagTracker,
    APRILTAG_POSITIONS,
    find_closest_tags,
    calculate_tag_centroid,
    calculate_pan_to_tags,
)

__all__ = [
    'CameraController',
    'ViewTracker',
    'TagDetector',
    'detect_markers',
    'draw_tag_detections',
    'DistanceMeasurement',
    'Triangulation',
    'KalmanFilter2D',
    'DynamicTagTracker',
    'APRILTAG_POSITIONS',
    'find_closest_tags',
    'calculate_tag_centroid',
    'calculate_pan_to_tags',
]

