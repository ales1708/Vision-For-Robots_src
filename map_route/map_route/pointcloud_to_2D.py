import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import binary_closing


def load_and_filter_points(file_path, height_threshold):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    points_filtered = points[
        (points[:, 2] > height_threshold[0]) &
        (points[:, 2] < height_threshold[1])
    ]

    return pcd, points_filtered


def create_grid(points, resolution):
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    x_min -= 1; x_max += 1
    y_min -= 1; y_max += 1

    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)

    occupancy_map = np.zeros((height, width), dtype=np.int8)
    return occupancy_map, (x_min, y_min), width, height


def project_points_to_grid(points, x_min, y_min, resolution, width, height):
    indices_x = ((points[:, 0] - x_min) / resolution).astype(int)
    indices_y = ((points[:, 1] - y_min) / resolution).astype(int)

    indices_x = np.clip(indices_x, 0, width - 1)
    indices_y = np.clip(indices_y, 0, height - 1)

    return indices_x, indices_y


def apply_closing(occupancy_map, closing_size):
    if closing_size > 0:
        y, x = np.ogrid[-closing_size:closing_size+1, -closing_size:closing_size+1]
        disk = x**2 + y**2 <= closing_size**2
        occupancy_map = binary_closing(occupancy_map, structure=disk).astype(np.int8)
    return occupancy_map


def save_occupancy_map(occupancy_map, resolution, method, closing_size, save_output):
    plt.figure(figsize=(10, 8))
    plt.imshow(occupancy_map, cmap='gray_r', origin='lower')

    title = f"{method} 2D Occupancy Map (Res: {resolution}m"
    if closing_size > 0:
        title += f", Closing: {closing_size})"
    else:
        title += ")"

    plt.title(title)
    plt.xlabel("X axis (cells)")
    plt.ylabel("Y axis (cells)")
    plt.colorbar(label='Occupancy (0=free, 1=occupied)')

    if save_output:
        plt.savefig(save_output, dpi=150, bbox_inches='tight')
        np.savez(save_output.rsplit('.', 1)[0] + '_data.npz',
                 occupancy_map=occupancy_map,
                 resolution=resolution)

    plt.show()

def create_occupancy_map(file_path, resolution=0.1, height_threshold=(0.3, 2.0),
                        use_voxels=False, closing_size=0, save_output=None):
    pcd, points_filtered = load_and_filter_points(file_path, height_threshold)

    if use_voxels:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=resolution)
        voxels = voxel_grid.get_voxels()

        if len(voxels) == 0:
            raise ValueError("No voxels created!")

        origin = voxel_grid.origin
        points_filtered = np.array([
            origin + (np.array(v.grid_index) * resolution)
            for v in voxels
        ])

        points_filtered = points_filtered[
            (points_filtered[:, 2] > height_threshold[0]) &
            (points_filtered[:, 2] < height_threshold[1])
        ]

    occupancy_map, (x_min, y_min), width, height = create_grid(points_filtered, resolution)
    indices_x, indices_y = project_points_to_grid(points_filtered, x_min, y_min,
                                                   resolution, width, height)
    occupancy_map[indices_y, indices_x] = 1
    occupancy_map = apply_closing(occupancy_map, closing_size)
    method = "Voxel-based" if use_voxels else "Point-based"
    save_occupancy_map(occupancy_map, resolution, method, closing_size, save_output)

    return occupancy_map, (x_min, y_min), resolution


def main():
    parser = argparse.ArgumentParser(description='Create 2D occupancy map from point cloud')
    parser.add_argument('input_file')
    parser.add_argument('-r', type=float, default=0.1)
    parser.add_argument('--min-height', type=float, default=0.3)
    parser.add_argument('--max-height', type=float, default=2.0)
    parser.add_argument('-o')
    parser.add_argument('-c', type=int, default=0)
    parser.add_argument('--voxel', action='store_true')
    args = parser.parse_args()

    occ_map, origin, res = create_occupancy_map(
        args.input_file, args.r, (args.min_height, args.max_height),
        args.voxel, args.c, args.o
    )
    print(f"Map: {occ_map.shape} cells, origin: {origin}, res: {res}m")


if __name__ == "__main__":
    main()
