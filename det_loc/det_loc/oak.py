import rclpy
from apriltag import apriltag
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from .utils.marker_detection_utils import (
    draw_detections,
    multi_scale_marker_detection,
)
from .utils.camera_calibration_utils import CameraCalibration
from .utils.localization_utils import distance_measure, triangulation_3p, triangulation_2p, KalmanFilter2D, get_rotation_rvec
from .utils.camera_panning_utils import ViewTracker
from .utils.rover_detection import rover_detection, overlap_bboxes
import message_filters



class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("rgbd_subscriber")

        self.br = CvBridge()

        # ---- message_filters subscribers (ROS 2 style) ----
        self.rgb_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            "/oak/rgb/image_raw/compressed",
        )

        self.depth_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            "/oak/stereo/image_raw/compressedDepth",
        )

        # ---- ApproximateTimeSynchronizer ----
        # queue_size: how many msgs to keep in buffer
        # slop: allowed time difference between topics (in seconds)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info("RGBDSubscriber initialized (RGB + depth sync)")

    # =========================================================
    #          Synced callback: RGB + Depth together
    # =========================================================
    def synced_callback(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
        # ---------- Decode RGB (JPEG) ----------
        np_rgb = np.frombuffer(rgb_msg.data, np.uint8)
        rgb_frame = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)

        if rgb_frame is None:
            self.get_logger().warn("RGB decode failed")
            return

        # ---------- Decode Depth (compressedDepth) ----------
        # format example: "16UC1; compressedDepth"
        depth_fmt, compr_type = depth_msg.format.split(";")
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()

        # 12-byte header before the PNG image
        depth_header_size = 12
        raw_depth_data = depth_msg.data[depth_header_size:]

        np_depth = np.frombuffer(raw_depth_data, np.uint8)
        depth_img = cv2.imdecode(np_depth, cv2.IMREAD_UNCHANGED)

        if depth_img is None:
            self.get_logger().error("cv2.imdecode returned None for depth")
            return

        # If depth image has 3 channels for some reason, take the first
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]

        # ---------- Convert depth to something viewable ----------
        depth_f = depth_img.astype(np.float32)

        # Optional: inspect min/max once to tune range
        # print("depth min/max:", float(depth_f.min()), float(depth_f.max()))

        # Clip range for visualization (tune for your setup)
        depth_f = np.clip(depth_f, 300, 5000)  # e.g. 0.3–5m or similar

        depth_vis = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # ---------- Now you have BOTH rgb_frame and depth_img ----------
        # You can do whatever processing you want here.
        # For demo, just show both.

        cv2.imshow("RGB", rgb_frame)
        cv2.imshow("Depth", depth_color)
        cv2.waitKey(1)

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