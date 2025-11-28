import sys
if sys.prefix == '/home/aaron/miniconda3/envs/ros2_py':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/aaron/Desktop/Vision-For-Robots_src/install/det_loc'
