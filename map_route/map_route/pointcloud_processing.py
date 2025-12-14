import argparse
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from tqdm import tqdm


def manual_crop(pcd, x_min, x_max, y_min, y_max, z_min, z_max):
    pts = np.asarray(pcd.points)
    mask = np.ones(len(pts), dtype=bool)

    if x_min is not None:
        mask &= pts[:, 0] >= x_min
    if x_max is not None:
        mask &= pts[:, 0] <= x_max
    if y_min is not None:
        mask &= pts[:, 1] >= y_min
    if y_max is not None:
        mask &= pts[:, 1] <= y_max
    if z_min is not None:
        mask &= pts[:, 2] >= z_min
    if z_max is not None:
        mask &= pts[:, 2] <= z_max

    return pcd.select_by_index(np.where(mask)[0])

def voxel_downsample(pcd, voxel_size: float):
    if voxel_size <= 0:
        return pcd
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd_down

def statistical_outlier_removal(pcd, nb_neighbors: int, std_ratio: float):
    if len(pcd.points) < nb_neighbors:
        return pcd

    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd_clean

def dbscan_keep_largest(pcd, eps: float, min_points: int):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    valid = labels[labels >= 0]
    counts = np.bincount(valid)
    largest_label = int(np.argmax(counts))
    idx = np.where(labels == largest_label)[0]

    return pcd.select_by_index(idx)


def keep_vertical_surfaces(pcd, normal_radius: float, normal_max_nn: int, vertical_nz: float):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn
        )
    )

    normals = np.asarray(pcd.normals)
    mask = np.abs(normals[:, 2]) <= vertical_nz
    return pcd.select_by_index(np.where(mask)[0])


class PointNetEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x: (B, N, 3)
        x = self.mlp(x)              # (B, N, D)
        x = torch.max(x, dim=1)[0]   # global max pool -> (B, D)
        return x


def pointnet_pipeline(pcd, dl_points: int, k: int, feature_dim: int,
                      cluster_eps: float, cluster_min: int):
    pts = np.asarray(pcd.points)
    N = min(dl_points, len(pts))

    # Subsample and build KD-tree
    dl_pts = pts[np.random.choice(len(pts), N, replace=False)]
    dl_pcd = o3d.geometry.PointCloud()
    dl_pcd.points = o3d.utility.Vector3dVector(dl_pts)
    dl_tree = o3d.geometry.KDTreeFlann(dl_pcd)

    # Extract features
    model = PointNetEncoder(feature_dim).eval()
    features = np.zeros((N, feature_dim), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(N), desc="  Extracting features"):
            _, nn_idx, _ = dl_tree.search_knn_vector_3d(dl_pts[i], k)
            patch = torch.from_numpy(dl_pts[nn_idx] - dl_pts[i]).float().unsqueeze(0)
            features[i] = model(patch).numpy()[0]

    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8

    # Cluster and find dominant cluster
    labels = DBSCAN(eps=cluster_eps, min_samples=cluster_min, n_jobs=1).fit(features).labels_
    valid = labels[labels >= 0]
    if valid.size == 0:
        return pcd

    main_cluster = np.bincount(valid).argmax()

    # Propagate labels to full cloud
    _, nn = cKDTree(dl_pts).query(pts, k=1)
    keep_idx = np.where(labels[nn] == main_cluster)[0]

    return pcd.select_by_index(keep_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--cluster_eps", type=float, default=0.12)
    ap.add_argument("--skip_vertical", action="store_true")
    ap.add_argument("--skip_pointnet", action="store_true")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.inp)
    if pcd.is_empty():
        raise SystemExit("ERROR: Empty point cloud")

    pcd = voxel_downsample(pcd, args.voxel)
    pcd = statistical_outlier_removal(pcd, nb_neighbors=30, std_ratio=1.5)
    pcd = dbscan_keep_largest(pcd, eps=args.cluster_eps, min_points=30)

    if not args.skip_vertical:
        pcd = keep_vertical_surfaces(pcd, normal_radius=0.08, normal_max_nn=40, vertical_nz=0.35)

    if not args.skip_pointnet:
        pcd = pointnet_pipeline(pcd, dl_points=20000, k=64, feature_dim=64,
                               cluster_eps=0.6, cluster_min=20)

    o3d.io.write_point_cloud(args.out, pcd, write_ascii=False, compressed=True)


if __name__ == "__main__":
    main()
