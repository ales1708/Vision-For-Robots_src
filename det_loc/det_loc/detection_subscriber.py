import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from apriltag import apriltag
import math

use_custom_edge_detector = False


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = None

    def update(self, error, current_time):
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0

        dt = (current_time - self.prev_time).nanoseconds / 1e9
        if dt <= 0:
            dt = 1e-3
        self.prev_time = current_time

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.br = CvBridge()
        self.detector = apriltag("tagStandard41h12")

        self.subscription = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.listener_callback,
            10,
        )

        self.joint_pub = self.create_publisher(
            JointState, "/ugv/joint_states", 10
        )

        # -------- camera scanning state --------
        self.drive_state = False               # True when ready to drive
        self.last_num_detections = 0

        # pan scan limits
        self.pan_min = -1.5
        self.pan_max =  1.5
        self.pan_step = 0.15
        self.pan_position = 0.0
        self.pan_direction = +1   # start sweeping right

        # track full sweep coverage
        self.reached_min = False
        self.reached_max = False

        # startup wait
        self.startup_wait_sec = 10.0
        self.start_time = self.get_clock().now()

        # scanning timer (no blocking)
        self.scan_timer = self.create_timer(
            0.4,   # 10 Hz scan step
            self.scanning_step
        )

    def publish_pan_tilt(self, pan, tilt=0.0):
        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        msg.position = [float(pan), float(tilt)]
        self.joint_pub.publish(msg)

    def scanning_step(self):
        # If already in drive mode → center & stop timer
        if self.drive_state:
            self.publish_pan_tilt(0.0)
            self.destroy_timer(self.scan_timer)
            return

        # ---- STARTUP WAIT ----
        now = self.get_clock().now()
        dt = (now - self.start_time).nanoseconds / 1e9
        if dt < self.startup_wait_sec:
            return    # keep waiting

        # ---- ENOUGH DETECTIONS? ----
        if self.last_num_detections >= 2:
            self.drive_state = True
            self.get_logger().info("Found detections → locking camera at center.")
            self.publish_pan_tilt(0.0)
            self.destroy_timer(self.scan_timer)
            return

        # ---- FULL SWEEP COVERED AND STILL NOTHING? ----
        if self.reached_min and self.reached_max:
            self.get_logger().info("Full sweep done → no detections → centering.")
            self.publish_pan_tilt(0.0)
            self.destroy_timer(self.scan_timer)
            return

        # ---- CONTINUE SWEEPING ----
        self.pan_position += self.pan_direction * self.pan_step

        # hit right boundary?
        if self.pan_position >= self.pan_max:
            self.pan_position = self.pan_max
            self.reached_max = True
            self.pan_direction = -1  # reverse direction

        # hit left boundary?
        if self.pan_position <= self.pan_min:
            self.pan_position = self.pan_min
            self.reached_min = True
            self.pan_direction = +1  # reverse direction

        self.publish_pan_tilt(self.pan_position)


    def listener_callback(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        result, num_detections = marker_detection(frame, self.detector)
        cv2.imshow("Marker Detection", frame)
        cv2.waitKey(1)


def remove_duplicate_detections(detections, distance_threshold=20):
    """
    Remove duplicate detections of the same tag at different scales.
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

def frame_processing_scale(gray_image, scale, use_new_processing=False):
    """
    Process frame at a specific scale.
    """
    image = cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    if use_new_processing:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image

def multi_scale_marker_detection(frame, detector, scales=[1.5, 2.0, 2.5], use_new_processing=False):
    """
    Detect markers at multiple scales and combine results.
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_detections = []

    for scale in scales:
        processed = frame_processing_scale(gray_image, scale, use_new_processing)
        detections = detector.detect(processed)

        # Adjust coordinates back to original scale
        for det in detections:
            det['center'] = (det['center'][0] / scale, det['center'][1] / scale)
            det['lb-rb-rt-lt'] = [(x / scale, y / scale) for x, y in det['lb-rb-rt-lt']]

        all_detections.extend(detections)

    # Remove duplicates
    unique_detections = remove_duplicate_detections(all_detections)
    result = frame
    if len(unique_detections) > 0:
        result = draw_detections(frame, unique_detections, scale=1.0)

    return result, len(unique_detections)

def frame_processing(current_frame):
    scale = 2.0
    image = cv2.resize(current_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    return image

def draw_detections(color_image, detections, scale=2.0):
    """
    Draw bounding boxes and tag IDs on the color image.

    Args:
        color_image: Original RGB image
        detections: List of AprilTag detections
        scale: Scale factor used during preprocessing (to adjust coordinates)
    """
    vis_image = color_image.copy()

    for detection in detections:
        corners = detection['lb-rb-rt-lt']
        corners = [(int(x/scale), int(y/scale)) for x, y in corners]
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i+1) % 4]
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)

        center = (int(detection['center'][0]/scale), int(detection['center'][1]/scale))
        cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
        tag_id = detection['id']
        cv2.putText(vis_image, f"ID: {tag_id}",
                    (center[0] - 20, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return vis_image

def marker_detection(frame, detector):
    result, num_detections = multi_scale_marker_detection(frame, detector, scales=[1.5, 2.0, 2.5], use_new_processing=True)

    return result, num_detections

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
