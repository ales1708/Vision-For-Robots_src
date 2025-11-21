#!/usr/bin/env python3
"""
Utility script to visualize AprilTag debug output files.
Converts PNM files to PNG and displays them in a grid.
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path


def convert_pnm_to_png(pnm_path, output_dir="debug_visualizations"):
    """Convert a PNM file to PNG format."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read PNM file
        image = cv2.imread(pnm_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Warning: Could not read {pnm_path}")
            return None

        # Generate output filename
        base_name = Path(pnm_path).stem
        output_path = os.path.join(output_dir, f"{base_name}.png")

        # Save as PNG
        cv2.imwrite(output_path, image)
        print(f"Converted: {pnm_path} -> {output_path}")

        return output_path
    except Exception as e:
        print(f"Error converting {pnm_path}: {e}")
        return None


def create_debug_grid(debug_files, output_path="debug_grid.png", max_width=1920):
    """
    Create a grid visualization of all debug images.

    Args:
        debug_files: List of image file paths
        output_path: Where to save the grid
        max_width: Maximum width for the output image
    """
    if not debug_files:
        print("No debug files to visualize")
        return

    # Read all images
    images = []
    titles = []

    for file_path in debug_files:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
            titles.append(Path(file_path).stem)

    if not images:
        print("No valid images found")
        return

    # Determine grid layout (2 columns)
    n_images = len(images)
    n_cols = 2
    n_rows = (n_images + n_cols - 1) // n_cols

    # Calculate dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Scale down if too large
    scale = min(1.0, max_width / (max_w * n_cols))
    cell_h = int(max_h * scale)
    cell_w = int(max_w * scale)

    # Add padding for titles
    title_height = 30
    cell_h_with_title = cell_h + title_height

    # Create canvas
    grid_h = cell_h_with_title * n_rows
    grid_w = cell_w * n_cols

    # Handle grayscale vs color
    if len(images[0].shape) == 2:
        grid = np.ones((grid_h, grid_w), dtype=np.uint8) * 255
    else:
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # Place images in grid
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // n_cols
        col = idx % n_cols

        # Resize image
        resized = cv2.resize(img, (cell_w, cell_h))

        # Convert grayscale to color if needed
        if len(grid.shape) == 3 and len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        elif len(grid.shape) == 2 and len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Calculate position
        y_start = row * cell_h_with_title + title_height
        y_end = y_start + cell_h
        x_start = col * cell_w
        x_end = x_start + cell_w

        # Place image
        grid[y_start:y_end, x_start:x_end] = resized

        # Add title
        title_y = row * cell_h_with_title + 20
        title_x = x_start + 10
        cv2.putText(grid, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(output_path, grid)
    print(f"Created debug grid: {output_path}")
    return output_path


def visualize_all_debug_files(directory=".", output_dir="debug_visualizations"):
    """
    Find and visualize all AprilTag debug files in a directory.
    """
    print("=" * 60)
    print("AprilTag Debug File Visualizer")
    print("=" * 60)

    # Find all debug PNM files
    pnm_files = glob.glob(os.path.join(directory, "debug_*.pnm"))

    if not pnm_files:
        print(f"No debug_*.pnm files found in {directory}")
        return

    print(f"\nFound {len(pnm_files)} debug files:")
    for f in pnm_files:
        print(f"  - {os.path.basename(f)}")

    # Convert all to PNG
    print("\nConverting PNM files to PNG...")
    png_files = []
    for pnm_file in pnm_files:
        png_path = convert_pnm_to_png(pnm_file, output_dir)
        if png_path:
            png_files.append(png_path)

    # Create grid visualization
    if png_files:
        print("\nCreating debug visualization grid...")
        grid_path = os.path.join(output_dir, "debug_grid.png")
        create_debug_grid(png_files, grid_path)
        print(f"\n✓ All visualizations saved to: {output_dir}/")
        print(f"✓ Grid visualization: {grid_path}")

    # Check for PostScript files
    ps_files = glob.glob(os.path.join(directory, "debug_*.ps"))
    if ps_files:
        print(f"\nFound {len(ps_files)} PostScript (.ps) files:")
        for f in ps_files:
            print(f"  - {os.path.basename(f)}")
        print("  Open with: open -a Preview debug_lines.ps")


def show_debug_info():
    """Display information about AprilTag debug files."""
    info = """
    AprilTag Debug Files:
    =====================

    debug_preprocess.pnm  - Preprocessed grayscale image
    debug_output.pnm      - Final detection output
    debug_quads_fixed.pnm - Detected quadrilaterals after refinement
    debug_clusters.pnm    - Connected component clusters
    debug_lines.ps        - Line detection visualization (PostScript)

    These files are generated when debug=True is set in the detector.
    """
    print(info)


if __name__ == "__main__":
    import sys

    # Get directory from command line or use current
    directory = sys.argv[1] if len(sys.argv) > 1 else "."

    show_debug_info()
    visualize_all_debug_files(directory)

    print("\n" + "=" * 60)
    print("To view individual files:")
    print("  open debug_visualizations/debug_preprocess.png")
    print("  open debug_visualizations/debug_grid.png")
    print("=" * 60)

