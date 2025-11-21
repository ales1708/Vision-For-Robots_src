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
        self.drive_state = False
        self.initial_camera_position_found = False
        self.pan_position = 0.0
        self.initial_camera_timer = self.create_timer(1.0, self.starting_camera_position)

        self.subscription = self.create_subscription(
            Image,  # use CompressedImage or Image
            "/image_raw",
            self.listener_callback,
            10,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.pid_controller = PIDController(Kp=1.0, Ki=0.2, Kd=0.1)
        self.br = CvBridge()
        self.detector = apriltag("tagStandard41h12")

    def starting_camera_position(self):
        """set camera position to look up"""

        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        msg.position = [-0.75, 0.0]  # - 45 degrees and 0 degrees in radians
        self.joint_pub.publish(msg)
        self.get_logger().info("Camera turned to preset position.")
        self.destroy_timer(self.initial_camera_timer)

    def driving_camera_position(self, num_detections):
        if num_detections >= 3:
            self.drive_state = True
            return

        if self.pan_position >= 0.75:
            self.pan_position = 0.0

            msg = JointState()
            msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
            msg.position = [0.0, 0.0]
            self.joint_pub.publish(msg)
            self.drive_state = True

            return

        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        self.pan_position += 0.25
        msg.position = [self.pan_position, 0.0]
        self.joint_pub.publish(msg)

        time.sleep(0.2)

    def listener_callback(self, data):
        """Uses the subscribed camera feed to detect lines and follow them"""
        frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8") # for Image
        result, num_detections = marker_detection(frame, self.detector)

        if not self.drive_state:
            self.driving_camera_position(num_detections)

        cv2.imshow("Marker Detection", result)
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
    draw_img = frame.copy()
    cv2.imshow("Original Image", draw_img)
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
