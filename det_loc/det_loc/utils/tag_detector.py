"""
AprilTag Detection Module

This module handles AprilTag detection with multi-scale processing,
duplicate removal, and visualization.
"""

import cv2
import math


class TagDetector:
    """
    AprilTag detector with multi-scale processing and duplicate removal.

    This class wraps the apriltag detector and provides enhanced detection
    capabilities including multi-scale processing for improved detection rates.
    """

    def __init__(self, detector, config=None):
        """
        Initialize tag detector.

        Args:
            detector: apriltag detector instance
            config: Optional dict with configuration parameters
        """
        self.detector = detector

        # Configuration
        cfg = config or {}
        self.scales = cfg.get("scales", [1.5, 2.0, 2.5])
        self.use_enhanced_processing = cfg.get("use_enhanced_processing", True)
        self.duplicate_threshold = cfg.get("duplicate_threshold", 20)
        self.draw_detections = cfg.get("draw_detections", True)

    def detect(self, frame):
        """
        Detect AprilTags in frame using multi-scale processing.

        Args:
            frame: BGR color image

        Returns:
            Tuple of (annotated_image, detections_list)
        """
        result, detections = self._multi_scale_detection(
            frame,
            self.scales,
            self.use_enhanced_processing
        )
        return result, detections

    def _process_frame_at_scale(self, gray_image, scale, use_enhanced=False):
        """
        Process frame at a specific scale with optional enhancement.

        Args:
            gray_image: Grayscale image
            scale: Scale factor for resizing
            use_enhanced: Whether to apply morphological operations

        Returns:
            Processed grayscale image
        """
        # Resize image
        image = cv2.resize(
            gray_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Optional morphological processing
        if use_enhanced:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return image

    def _remove_duplicates(self, detections):
        """
        Remove duplicate detections of the same tag at different scales.

        Args:
            detections: List of detection dicts

        Returns:
            List of unique detections (keeps best quality)
        """
        if len(detections) == 0:
            return []

        unique_detections = []

        for det in detections:
            is_duplicate = False
            for unique_det in unique_detections:
                if det['id'] == unique_det['id']:
                    # Calculate distance between centers
                    dx = det['center'][0] - unique_det['center'][0]
                    dy = det['center'][1] - unique_det['center'][1]
                    distance = math.sqrt(dx**2 + dy**2)

                    if distance < self.duplicate_threshold:
                        # It's a duplicate, keep the one with better hamming score
                        if det['hamming'] < unique_det['hamming']:
                            unique_detections.remove(unique_det)
                            unique_detections.append(det)
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_detections.append(det)

        return unique_detections

    def _multi_scale_detection(self, frame, scales, use_enhanced=False):
        """
        Detect markers at multiple scales and combine results.

        Args:
            frame: BGR color image
            scales: List of scale factors to try
            use_enhanced: Whether to use enhanced processing

        Returns:
            Tuple of (annotated_image, unique_detections)
        """
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_detections = []

        for scale in scales:
            processed = self._process_frame_at_scale(gray_image, scale, use_enhanced)
            detections = self.detector.detect(processed)

            # Adjust coordinates back to original scale
            for det in detections:
                det['center'] = (
                    det['center'][0] / scale,
                    det['center'][1] / scale
                )
                det['lb-rb-rt-lt'] = [
                    (x / scale, y / scale)
                    for x, y in det['lb-rb-rt-lt']
                ]

            all_detections.extend(detections)

        # Remove duplicates
        unique_detections = self._remove_duplicates(all_detections)

        # Draw detections if enabled
        result = frame
        if len(unique_detections) > 0 and self.draw_detections:
            result = draw_tag_detections(frame, unique_detections, scale=1.0)

        return result, unique_detections


def draw_tag_detections(color_image, detections, scale=1.0):
    """
    Draw bounding boxes and tag IDs on the color image.

    Args:
        color_image: Original BGR image
        detections: List of AprilTag detections
        scale: Scale factor used during preprocessing (to adjust coordinates)

    Returns:
        Annotated image
    """
    vis_image = color_image.copy()

    for detection in detections:
        # Draw bounding box
        corners = detection['lb-rb-rt-lt']
        corners = [(int(x / scale), int(y / scale)) for x, y in corners]

        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i + 1) % 4]
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)

        # Draw center point
        center = (
            int(detection['center'][0] / scale),
            int(detection['center'][1] / scale)
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
            2
        )

    return vis_image


# Convenience function for backward compatibility
def detect_markers(frame, detector, scales=None, use_enhanced=True):
    """
    Convenience function to detect markers with default settings.

    Args:
        frame: BGR color image
        detector: apriltag detector instance
        scales: Optional list of scale factors (default: [1.5, 2.0, 2.5])
        use_enhanced: Whether to use enhanced processing

    Returns:
        Tuple of (annotated_image, detections_list)
    """
    if scales is None:
        scales = [1.5, 2.0, 2.5]

    config = {
        "scales": scales,
        "use_enhanced_processing": use_enhanced,
        "draw_detections": True,
    }

    tag_detector = TagDetector(detector, config)
    return tag_detector.detect(frame)

