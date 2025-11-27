from nav_msgs.msg import Odometry
import math
from sensor_msgs.msg import Imu

self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)

def imu_callback(self, msg):
    q0 = msg.orientation.w
    q1 = msg.orientation.x
    q2 = msg.orientation.y
    q3 = msg.orientation.z

    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    self.imu_yaw = math.atan2(siny_cosp, cosy_cosp)

class YourNode(Node):
    def __init__(self):
        super().__init__('your_node')
        self.yaw = 0.0
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        # Extract quaternion from odometry pose
        q0 = msg.pose.pose.orientation.w
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z

        # Convert quaternion to yaw (in radians)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)