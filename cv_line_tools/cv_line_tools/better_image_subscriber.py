import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
import cv2
import numpy as np
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
        self.last_line = None
        self.initial_camera_timer = self.create_timer(1.0, self.initial_camera_position)
        self.in_rosbag = True
        self.angular_speed_list = []

    def initial_camera_position(self):
        """set camera position to look up"""

        msg = JointState()
        msg.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        msg.position = [0.0, 1.57]  # 0 degrees and 90 degrees in radians
        self.joint_pub.publish(msg)
        self.get_logger().info("Camera turned to preset position.")
        self.destroy_timer(self.initial_camera_timer)

    def listener_callback(self, data):
        """Uses the subscribed camera feed to detect lines and follow them"""

        frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8") # for Image
        # np_arr = np.frombuffer(data.data, np.uint8)  # for CompressedImage
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # for CompressedImage

        # detect ceiling light lines in camera frame
        line = line_detector(
            frame,
            line_detector_type="canny",
            last_line=self.last_line,
            in_rosbag=self.in_rosbag,
        )

        # react if there are any lines detected
        if line is not None:
            z_error = speed_controller(line)
            control = self.pid_controller.update(z_error, self.get_clock().now())

            twist = Twist()

            # if no corrections are needed, drive forward
            # otherwise turn to correct the angle
            if z_error < 10 and z_error > -10:
                twist.linear.x = 0.3
                twist.angular.z = 0.0

            else:
                twist.linear.x = 0.0

                if z_error > 10:
                    twist.angular.z = 0.3
                else:
                    twist.angular.z = -0.3
                print(f"z_error: {z_error}, control: {control}")

            self.angular_speed_list.append(float(np.clip(-control, -1.0, 1.0)))
            self.last_line = line

        else:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.last_line = None

        self.cmd_vel_pub.publish(twist)

        cv2.waitKey(1)


def frame_processing(current_frame):
    """applies preprocessing on an image to return a binary image of the ceilin lights
    current_frame: current frame as an RGB image"""

    bright_percentile = 98
    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    # create global brightness map
    t_glob = np.percentile(v, bright_percentile)
    mask_glob = (v >= t_glob).astype(np.uint8) * 255

    # mask to select low saturation pixels
    mask_white = cv2.inRange(s, 0, 40)

    light_mask = cv2.bitwise_and(mask_glob, mask_white)

    # morphological closing and dilations
    light_mask = cv2.morphologyEx(
        light_mask, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8), iterations=1
    )
    light_mask = cv2.dilate(light_mask, np.ones((3, 3), np.uint8), iterations=1)

    return light_mask


def line_segment_detector(current_frame):
    """detects lines in an image using the LSD algorithm"""

    # get binary image of bright withis areas
    light_mask = frame_processing(current_frame)

    # use LSD to detect lines in the binary image
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(light_mask)[0]

    return lines


def canny_edge_detector(current_frame, blur_ksize=5, sigma=0.45, use_L2=True):
    """detects edges in an image using the canny edge detection algorihthm"""

    # get smoothed grayscale image for Canny edge detection
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray_blur = gray

    # get binary image of bright withis areas
    light_mask = frame_processing(current_frame)

    # use Canny edge detection on the binary image
    v_med = np.median(gray_blur)
    lower = int(max(0, (1.0 - sigma) * v_med))
    upper = int(min(255, (1.0 + sigma) * v_med))
    edges = cv2.Canny(light_mask, lower, upper, L2gradient=use_L2)

    # morphological closing of detected edges
    edges = cv2.bitwise_and(edges, light_mask)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )

    return edges


def line_measures(line):
    """measures the midpoint and angle of a line"""

    x1, y1, x2, y2 = line
    mx, my = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    angle = math.atan2((y2 - y1), (x2 - x1))
    return mx, my, angle


def line_processor(
    lines,
    img,
    last_line=None,
    max_pos_dist_sq=200**2,
    max_ang_dist_deg=20,
):
    """
    Processes a list of lines to return the best line to follow. Lines are select based on last line, or the line closest to the image center.
    """

    # get image center
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # normalize lines to a simple list of (x1,y1,x2,y2)
    norm_lines = [
        (float(x1), float(y1), float(x2), float(y2))
        for x1, y1, x2, y2 in [l[0] for l in lines]
    ]

    if last_line is not None:
        # get the midpoint and angle of the last line
        lmx, lmy, lang = line_measures(last_line)

        # initialize the best score and candidate
        best_score = 1e9
        best_candidate = None

        # compare each line to the best candidate line
        for x1, y1, x2, y2 in norm_lines:
            mx, my, ang = line_measures((x1, y1, x2, y2))
            pos_dist_sq = (mx - lmx) ** 2 + (my - lmy) ** 2

            ang_diff = abs(math.degrees(ang - lang))
            if ang_diff > 180:
                ang_diff = 360 - ang_diff

            score =  pos_dist_sq +  (ang_diff**2)
            if score < best_score:
                best_score = score
                best_candidate = (int(x1), int(y1), int(x2), int(y2))
                best_pos_dist_sq = pos_dist_sq
                best_ang_diff = ang_diff

        # check if this best candidate is close enough
        if best_candidate is not None:
            close_in_pos = best_pos_dist_sq <= max_pos_dist_sq
            close_in_ang = best_ang_diff <= max_ang_dist_deg
            if close_in_pos and close_in_ang:
                return best_candidate

    # If the best candidate is not close enough we fall back to looking in the center
    best_dist = 1e9
    center_best = None
    for x1, y1, x2, y2 in norm_lines:
        mx, my, ang = line_measures((x1, y1, x2, y2))
        dist = (mx - cx) ** 2 + (my - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            center_best = (int(x1), int(y1), int(x2), int(y2))

    return center_best


def angle_diff(a, b):
    """calculates the difference in angles a and b"""

    d = a - b
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def speed_controller(follow_line):
    """measures the angular deviation of the detected line from vertical
    and returns that as an error value for steering correction."""

    x1, y1, x2, y2 = follow_line
    dy = y2 - y1
    dx = x2 - x1
    theta = math.degrees(math.atan2(dy, dx))

    theta_targets = [90, -90]  # accept either vertical
    target = min(theta_targets, key=lambda t: abs(angle_diff(theta, t)))

    z_error = angle_diff(theta, target)

    # z_error = error / 180  # range [-1, 1] to prevent huge errors
    return z_error


def line_detector(
    current_frame,
    line_detector_type="canny",
    last_line=None,
    in_rosbag=False,
    min_len=25,
):
    """Detects lines in the current frame following either the
    canny edge detector or LSD algorithm, then picks a target line to follow.

    line_detector_type: "canny" or lsd"""

    draw_img = current_frame.copy()
    process_frame = current_frame

    if in_rosbag:
        process_frame = process_frame[: process_frame.shape[0] // 4, :]

    # detect edges in the current frame with Canny edge detection
    if line_detector_type.lower() == "canny":
        edges = canny_edge_detector(
            current_frame=process_frame, blur_ksize=7, sigma=0.5
        )

        # apply Hough transformation to get lines from the edges
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=15,
            maxLineGap=5,
        )

    # detect lines in the current frame with the LSD algorithm
    elif line_detector_type.lower() == "lsd":
        raw = line_segment_detector(process_frame)

    # convert raw line detections into a list of line endpoints as floats
    detected = [
        (float(x1), float(y1), float(x2), float(y2))
        for l in (raw if raw is not None else [])
        for (x1, y1, x2, y2) in [l[0]]
    ]

    # filter out short line segments
    filtered = [
        [[x1, y1, x2, y2]]
        for x1, y1, x2, y2 in detected
        if math.hypot(x2 - x1, y2 - y1) >= min_len
    ]

    lines = np.array(filtered, dtype=np.float32) if filtered else None

    # select a line to follow from all detected lines
    follow_line = None
    if lines is not None and len(lines):
        for ((x1, y1, x2, y2),) in lines:
            pt1 = (int(round(x1)), int(round(y1)))
            pt2 = (int(round(x2)), int(round(y2)))
            cv2.line(draw_img, pt1, pt2, (0, 255, 0), 2)

        follow_line = line_processor(lines, current_frame, last_line)
        if follow_line is not None:
            x1, y1, x2, y2 = follow_line
            cv2.line(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    else:
        print("No lines detected")

    cv2.imshow("image", draw_img)
    return follow_line


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
