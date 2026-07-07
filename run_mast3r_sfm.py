import sys
import os
import glob
import torch
import numpy as np

# --- CONFIGURATION ---
MAST3R_PATH = "/teamspace/studios/this_studio/mast3r"
IMAGE_DIR = "/teamspace/studios/this_studio/test_data/Cabinet"
OUTPUT_DIR = "/teamspace/studios/this_studio/cabinet_output"
CONF_THRESHOLD = 1.0  # Strict/clean
# ---------------------

sys.path.insert(0, MAST3R_PATH)

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from dust3r.utils.image import load_images

def main():
    # 1. Load Model
    print("Loading MASt3R model...")
    device = "cuda"
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)

    # 2. Load Images
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    print(f"Found {len(image_paths)} images")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")
    
    # Load with fixed size (512 edge)
    imgs = load_images(image_paths, size=512, verbose=True)

    # 3. Create Pairs (Complete Graph)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} pairs (complete graph)")

    # 4. Run Optimization
    cache_path = os.path.join(OUTPUT_DIR, "cache")
    os.makedirs(cache_path, exist_ok=True)

    print("Running sparse_global_alignment...")
    scene = sparse_global_alignment(
        image_paths, 
        pairs, 
        cache_path, 
        model,
        lr1=0.07, niter1=500,
        lr2=0.014, niter2=200,
        matching_conf_thr=CONF_THRESHOLD, 
        shared_intrinsics=True,
        subsample=8,
        device=device
    )

    # 5. Extract & Save Results
    print("Extracting results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Poses & Intrinsics
    poses = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    pps = scene.get_principal_points().detach().cpu().numpy()

    np.save(os.path.join(OUTPUT_DIR, "camera_poses.npy"), poses.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "camera_focals.npy"), focals.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "camera_pps.npy"), pps.astype(np.float32))
    print(f"Saved poses: {poses.shape}")

    # Dense Point Cloud
    pts3d_list, colors_list, confs_list = scene.get_dense_pts3d()
    
    all_pts, all_colors, all_confs = [], [], []

    for i, (pts, conf) in enumerate(zip(pts3d_list, confs_list)):
        pts_np = pts.reshape(-1, 3).cpu().numpy()
        conf_np = conf.reshape(-1).cpu().numpy()
        
        img_tensor = imgs[i]["img"]
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 0.5 + 0.5)
        colors_np = img_np.reshape(-1, 3)
        
        all_pts.append(pts_np)
        all_colors.append(colors_np)
        all_confs.append(conf_np)

    pts3d = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    confs = np.concatenate(all_confs, axis=0)

    # Filter
    mask = confs >= CONF_THRESHOLD
    pts3d_filtered = pts3d[mask]
    colors_filtered = (np.clip(colors[mask], 0, 1) * 255).astype(np.uint8)

    print(f"Total points: {len(pts3d)}, Filtered (conf>={CONF_THRESHOLD}): {len(pts3d_filtered)}")

    # Write PLY
    from plyfile import PlyData, PlyElement
    vertices = np.zeros(len(pts3d_filtered), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    vertices["x"] = pts3d_filtered[:, 0]
    vertices["y"] = pts3d_filtered[:, 1]
    vertices["z"] = pts3d_filtered[:, 2]
    vertices["red"] = colors_filtered[:, 0]
    vertices["green"] = colors_filtered[:, 1]
    vertices["blue"] = colors_filtered[:, 2]

    ply_path = os.path.join(OUTPUT_DIR, "pointcloud.ply")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(ply_path)
    print(f"Saved pointcloud to {ply_path}")
    print("Done!")

if __name__ == "__main__":
    main()
