"""
Marker Detection Utility Module

This module provides utility functions for AprilTag marker detection,
including multi-scale detection, preprocessing, and visualization.
"""

import cv2
import numpy as np
import math
from apriltag import apriltag


def remove_duplicate_detections(detections, distance_threshold=20):
    """
    Remove duplicate detections of the same tag at different scales.

    Args:
        detections: List of AprilTag detections
        distance_threshold: Maximum distance between centers to consider duplicates

    Returns:
        List of unique detections (keeping the one with best hamming score)
    """
    if len(detections) == 0:
        return []

    unique_detections = []

    for det in detections:
        is_duplicate = False
        for unique_det in unique_detections:
            if det['id'] == unique_det['id']:
                dx = det['center'][0] - unique_det['center'][0]
                dy = det['center'][1] - unique_det['center'][1]
                distance = math.sqrt(dx**2 + dy**2)

                if distance < distance_threshold:
                    # It's a duplicate, keep the one with higher hamming score
                    if det['hamming'] < unique_det['hamming']:
                        unique_detections.remove(unique_det)
                        unique_detections.append(det)
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_detections.append(det)

    return unique_detections


def frame_processing_scale(gray_image, scale, use_clahe=True, use_morphology=False):
    """
    Process frame at a specific scale with optional enhancements.

    Args:
        gray_image: Grayscale input image
        scale: Scale factor for resizing
        use_clahe: Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        use_morphology: Whether to apply morphological operations

    Returns:
        Processed image at the specified scale
    """
    image = cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    if use_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image


def preprocess_image(color_image, scale=2.0, use_clahe=True):
    """
    Preprocess color image for AprilTag detection.

    Args:
        color_image: Input BGR color image
        scale: Scale factor for resizing
        use_clahe: Whether to apply CLAHE

    Returns:
        Tuple of (processed_image, scale)
    """
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    return image, scale


def multi_scale_marker_detection(frame, detector, scales=[1.5, 2.0, 2.5],
                                 use_clahe=True, use_morphology=False):
    """
    Detect markers at multiple scales and combine results.

    Args:
        frame: Input BGR color image
        detector: AprilTag detector instance
        scales: List of scale factors to try
        use_clahe: Whether to apply CLAHE
        use_morphology: Whether to apply morphological operations

    Returns:
        Tuple of (all_detections, num_detections) where all_detections is a list
        of unique detections with coordinates adjusted to original scale
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_detections = []

    for scale in scales:
        processed = frame_processing_scale(gray_image, scale, use_clahe, use_morphology)
        detections = detector.detect(processed)

        # Adjust coordinates back to original scale
        for det in detections:
            det['center'] = (det['center'][0] / scale, det['center'][1] / scale)
            det['lb-rb-rt-lt'] = [(x / scale, y / scale) for x, y in det['lb-rb-rt-lt']]

        all_detections.extend(detections)

    # Remove duplicates
    unique_detections = remove_duplicate_detections(all_detections)

    return unique_detections, len(unique_detections)


def draw_detections(color_image, detections, scale=1.0):
    """
    Draw bounding boxes and tag IDs on the color image.

    Args:
        color_image: Original BGR color image
        detections: List of AprilTag detections
        scale: Scale factor used during preprocessing (to adjust coordinates)

    Returns:
        Image with detections drawn
    """
    vis_image = color_image.copy()

    for detection in detections:
        # Get corner coordinates and scale them back to original image size
        corners = detection['lb-rb-rt-lt']
        corners = [(int(x / scale), int(y / scale)) for x, y in corners]

        # Draw bounding box
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i + 1) % 4]
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)

        # Draw center point
        center = (
            int(detection['center'][0] / scale),
            int(detection['center'][1] / scale),
        )
        cv2.circle(vis_image, center, 5, (0, 0, 255), -1)

        # Draw tag ID
        tag_id = detection['id']
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


def marker_detection(frame, detector, multi_scale=True, scales=[1.5, 2.0, 2.5],
                     use_clahe=True, use_morphology=False):
    """
    Main marker detection function with optional multi-scale detection.

    Args:
        frame: Input BGR color image
        detector: AprilTag detector instance
        multi_scale: Whether to use multi-scale detection
        scales: List of scale factors for multi-scale detection
        use_clahe: Whether to apply CLAHE
        use_morphology: Whether to apply morphological operations

    Returns:
        Tuple of (detections, num_detections)
    """
    if multi_scale:
        detections, num_detections = multi_scale_marker_detection(
            frame, detector, scales, use_clahe, use_morphology
        )
    else:
        processed_frame, scale = preprocess_image(frame, scale=2.0, use_clahe=use_clahe)
        detections = detector.detect(processed_frame)
        # Adjust coordinates back to original scale
        for det in detections:
            det['center'] = (det['center'][0] / scale, det['center'][1] / scale)
            det['lb-rb-rt-lt'] = [(x / scale, y / scale) for x, y in det['lb-rb-rt-lt']]
        num_detections = len(detections)

    return detections, num_detections


def create_detector(tag_family="tagStandard41h12"):
    """
    Create and return an AprilTag detector instance.

    Args:
        tag_family: AprilTag family to detect

    Returns:
        AprilTag detector instance
    """
    return apriltag(tag_family)

