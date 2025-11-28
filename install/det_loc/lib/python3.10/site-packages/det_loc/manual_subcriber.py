import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
import cv2
import numpy as np
from apriltag import apriltag
import math
import os

class ImageSubscriber(Node):
    def __init__(self, use_multiscale=True):
        super().__init__("image_subscriber")

        self.subscription = self.create_subscription(
            Image,  # use CompressedImage or Image
            "/image_raw",
            self.listener_callback,
            10,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.joint_pub = self.create_publisher(JointState, "/ugv/joint_states", 10)
        self.br = CvBridge()
        self.detector = apriltag("tagStandard41h12")
        self.use_multiscale = use_multiscale

    def listener_callback(self, data):
        """Uses the subscribed camera feed to detect markers and follow them"""

        frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8") # for Image

        if self.use_multiscale:
            result, num_detections = multi_scale_marker_detection(
                frame, self.detector, scales=[1.5, 2.0, 2.5], use_new_processing=True
            )
        else:
            result, num_detections = marker_detection(frame, self.detector, old_processing=False)


def new_frame_processing(current_frame, aggressive=False):
    scale = 2.0
    image = cv2.resize(current_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image


def old_frame_processing(current_frame):
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


def marker_detection(frame, detector, old_processing=True):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if old_processing:
        processed_gray = old_frame_processing(gray_image)
    else:
        processed_gray = new_frame_processing(gray_image, aggressive=True)
    detections = detector.detect(processed_gray)
    result = frame
    if len(detections) >= 0:
        result = draw_detections(frame, detections, scale=2.0)

    return result, len(detections)


def marker_detection_test(detector, save_debug_per_image=True):
    """
    Test marker detection on multiple images with different processing methods.

    Args:
        detector: AprilTag detector instance
        save_debug_per_image: If True, save debug visualizations for each image in separate folders
    """
    # image_names = ['field1.png', 'field2.png', 'field3.png', 'field4.png', 'field5.png', 'field6.png', 'field7.png', 'field8.png']
    # image names should be all the images in the ../test_images/curling_test_imgs/ directory
    image_names = os.listdir("../../../test_images/curling_test_imgs")

    num_detections_old = []
    num_detections_new = []
    num_detections_multiscale = []

    print("\n" + "="*60)
    print("Running Marker Detection Tests")
    print("="*60 + "\n")

    for idx, image_name in enumerate(image_names, 1):
        print(f"[{idx}/{len(image_names)}] Processing {image_name}...")
        image = cv2.imread(f"../../../test_images/curling_test_imgs/{image_name}")

        if image is None:
            print(f"  ✗ Could not load {image_name}")
            continue

        # Run old processing (without debug to avoid conflicts)
        result_old, num_detections_old_i = marker_detection(image, detector, old_processing=True)
        num_detections_old.append(num_detections_old_i)

        # Run new processing (without debug)
        result_new, num_detections_new_i = marker_detection(image, detector, old_processing=False)
        num_detections_new.append(num_detections_new_i)

        # Run multi-scale detection (with debug enabled - this will generate debug files)
        result_multiscale, num_detections_multiscale_i = multi_scale_marker_detection(
            image, detector, scales=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0], use_new_processing=True
        )
        num_detections_multiscale.append(num_detections_multiscale_i)

        print(f"  Detections - Old: {num_detections_old_i}, New: {num_detections_new_i}, Multi-scale: {num_detections_multiscale_i}")

        # Stack all three images horizontally for comparison
        stacked_image = np.hstack((result_old, result_new, result_multiscale))

        # Add labels to distinguish each method
        label_y = 30
        cv2.putText(stacked_image, "Old Processing", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(stacked_image, "New Processing", (image.shape[1] + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(stacked_image, "Multi-Scale", (2 * image.shape[1] + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(f"res_imgs/new_imgs/stacked_{image_name}", stacked_image)

        # Move and organize debug files for this image
        if save_debug_per_image:
            debug_dir = move_and_convert_debug_files(image_name)
            if debug_dir:
                # Create a grid visualization for this image's debug files
                grid_path = create_debug_grid_for_image(debug_dir)
                if grid_path:
                    print(f"  ✓ Debug grid created: {grid_path}")
                print(f"  ✓ Debug files saved to: {debug_dir}")

    print("\n" + "="*60)
    print("Detection Summary")
    print("="*60)
    print(f"Number of detections (old):        {num_detections_old}")
    print(f"Number of detections (new):        {num_detections_new}")
    print(f"Number of detections (multi-scale): {num_detections_multiscale}")

    print(f"\nTotal detections:")
    print(f"  Old:         {sum(num_detections_old)}")
    print(f"  New:         {sum(num_detections_new)}")
    print(f"  Multi-scale: {sum(num_detections_multiscale)}")
    print("="*60 + "\n")
    for i in range(len(num_detections_old)):
        print(f"{image_names[i]}, {num_detections_old[i]}, {num_detections_new[i]}, {num_detections_multiscale[i]}")
    print("="*60 + "\n")

def move_and_convert_debug_files(image_name, base_output_dir="debug_visualizations"):
    """
    Move debug files for a specific image to its own subfolder and convert to PNG.

    Args:
        image_name: Name of the test image (e.g., 'field1.png')
        base_output_dir: Base directory for debug visualizations
    """
    import glob
    import os
    import shutil

    # Create subfolder for this image
    image_base = image_name.replace('.png', '')
    output_dir = os.path.join(base_output_dir, image_base)
    os.makedirs(output_dir, exist_ok=True)

    # Find all debug files
    debug_pnm_files = glob.glob("debug_*.pnm")
    debug_ps_files = glob.glob("debug_*.ps")

    if not debug_pnm_files and not debug_ps_files:
        print(f"  No debug files found for {image_name}")
        return output_dir

    # Move and convert PNM files
    for pnm_file in debug_pnm_files:
        try:
            # Read and convert to PNG
            img = cv2.imread(pnm_file)
            if img is not None:
                png_name = pnm_file.replace('.pnm', '.png')
                output_path = os.path.join(output_dir, png_name)
                cv2.imwrite(output_path, img)

            # Also keep original PNM
            pnm_dest = os.path.join(output_dir, pnm_file)
            shutil.move(pnm_file, pnm_dest)
        except Exception as e:
            print(f"  Error processing {pnm_file}: {e}")

    # Move PostScript files
    for ps_file in debug_ps_files:
        try:
            ps_dest = os.path.join(output_dir, ps_file)
            shutil.move(ps_file, ps_dest)
        except Exception as e:
            print(f"  Error moving {ps_file}: {e}")

    return output_dir


def create_debug_grid_for_image(debug_dir, output_filename="debug_grid.png"):
    """
    Create a grid visualization of all debug images for a single test image.

    Args:
        debug_dir: Directory containing debug PNG files
        output_filename: Name for the output grid image
    """
    import glob
    import os
    from pathlib import Path

    # Find all debug PNG files (excluding the grid itself)
    pattern = os.path.join(debug_dir, "debug_*.png")
    png_files = [f for f in glob.glob(pattern) if not f.endswith('debug_grid.png')]

    if not png_files:
        return None

    # Sort files in a logical order
    order = ['preprocess', 'threshold', 'segmentation', 'clusters',
             'samples', 'quads_raw', 'quads_fixed', 'output']

    def sort_key(filepath):
        basename = Path(filepath).stem
        for idx, keyword in enumerate(order):
            if keyword in basename:
                return idx
        return len(order)

    png_files.sort(key=sort_key)

    # Read all images
    images = []
    titles = []

    for file_path in png_files:
        img = cv2.imread(file_path)
        if img is not None:
            images.append(img)
            # Clean up title
            title = Path(file_path).stem.replace('debug_', '').replace('_', ' ').title()
            titles.append(title)

    if not images:
        return None

    # Determine grid layout (2 columns)
    n_images = len(images)
    n_cols = 2
    n_rows = (n_images + n_cols - 1) // n_cols

    # Calculate dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Scale down if too large
    max_width = 1920
    scale = min(1.0, max_width / (max_w * n_cols))
    cell_h = int(max_h * scale)
    cell_w = int(max_w * scale)

    # Add padding for titles
    title_height = 40
    cell_h_with_title = cell_h + title_height

    # Create canvas
    grid_h = cell_h_with_title * n_rows
    grid_w = cell_w * n_cols
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # Place images in grid
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // n_cols
        col = idx % n_cols

        # Resize image
        resized = cv2.resize(img, (cell_w, cell_h))

        # Convert grayscale to color if needed
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        # Calculate position
        y_start = row * cell_h_with_title + title_height
        y_end = y_start + cell_h
        x_start = col * cell_w
        x_end = x_start + cell_w

        # Place image
        grid[y_start:y_end, x_start:x_end] = resized

        # Add title
        title_y = row * cell_h_with_title + 25
        title_x = x_start + 10
        cv2.putText(grid, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Save grid
    output_path = os.path.join(debug_dir, output_filename)
    cv2.imwrite(output_path, grid)
    return output_path


def main(args=None):
    """
    Main function to run marker detection tests with debug visualizations.
    """
    # Create detector with debug enabled
    # detector = apriltag("tagStandard41h12", debug=False)
    detector = apriltag("tagStandard41h12", debug=False, refine_edges=True)

    # Run tests - debug files will be automatically organized per image
    marker_detection_test(detector, save_debug_per_image=False)

    print("\n✓ All tests complete!")
    print("  - Comparison images saved to: res_imgs/")
    print("  - Debug visualizations saved to: debug_visualizations/<image_name>/")

if __name__ == "__main__":
    main()