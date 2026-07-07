#!/usr/bin/env python3
"""
Speedy MASt3R - Quality Run
- 4fps with blur filtering
- shared_intrinsics=False  
- Outputs at conf 0.1, 0.5, 1.0
"""
import sys
import os
import glob
import torch
import numpy as np
import cv2
import time

SPEEDY_PATH = "/teamspace/studios/this_studio/speedy_mast3r"
IMAGE_DIR = "/teamspace/studios/this_studio/images"
OUTPUT_DIR = "/teamspace/studios/this_studio/output"
NITER1 = 500
NITER2 = 200
BLUR_THRESHOLD = 30  # Laplacian variance threshold (lower = blurrier)

sys.path.insert(0, SPEEDY_PATH)


def compute_blur_score(img_path):
    """Laplacian variance - higher is sharper"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def filter_blurry_images(image_paths, threshold=BLUR_THRESHOLD):
    """Remove images below blur threshold"""
    print(f"\n[Blur Filter] Analyzing {len(image_paths)} images...")
    scores = []
    for p in image_paths:
        score = compute_blur_score(p)
        scores.append((p, score))
    
    # Filter
    filtered = [(p, s) for p, s in scores if s >= threshold]
    removed = len(scores) - len(filtered)
    
    if filtered:
        avg_score = sum(s for _, s in filtered) / len(filtered)
        print(f"  Blur scores: min={min(s for _, s in scores):.0f}, max={max(s for _, s in scores):.0f}, avg={avg_score:.0f}")
        print(f"  Threshold: {threshold}")
        print(f"  Kept: {len(filtered)}, Removed: {removed}")
    
    return [p for p, _ in filtered]


def save_ply(pts3d, colors, conf_threshold, output_path):
    """Save point cloud with given confidence threshold"""
    from plyfile import PlyData, PlyElement
    
    mask = pts3d[:, 3] >= conf_threshold  # 4th column is confidence
    pts = pts3d[mask, :3]
    cols = colors[mask]
    
    if len(pts) == 0:
        print(f"  [WARN] No points at conf>={conf_threshold}")
        return 0
    
    vertices = np.zeros(len(pts), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    vertices["x"] = pts[:, 0]
    vertices["y"] = pts[:, 1]
    vertices["z"] = pts[:, 2]
    vertices["red"] = cols[:, 0]
    vertices["green"] = cols[:, 1]
    vertices["blue"] = cols[:, 2]
    
    PlyData([PlyElement.describe(vertices, "vertex")]).write(output_path)
    return len(pts)


def main():
    print("=" * 60)
    print("SPEEDY MAST3R - QUALITY RUN (4fps + blur filter)")
    print("=" * 60)
    
    device = "cuda"
    total_start = time.time()
    
    # 1. Load Model
    print("\n[Step 1/5] Loading model...")
    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)
    print("  Model loaded")
    
    # 2. Get Images
    print("\n[Step 2/5] Loading images...")
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    print(f"  Found {len(image_paths)} images")
    
    # 3. Blur Filter
    print("\n[Step 3/5] Filtering blurry frames...")
    image_paths = filter_blurry_images(image_paths, BLUR_THRESHOLD)
    if len(image_paths) < 5:
        print("ERROR: Not enough sharp images!")
        return
    
    imgs = load_images(image_paths, size=512, verbose=True)
    
    # 4. Create Pairs & Run SfM
    print("\n[Step 4/5] Running SfM...")
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    print(f"  Pairs: {len(pairs)}")
    
    cache_path = os.path.join(OUTPUT_DIR, "cache")
    os.makedirs(cache_path, exist_ok=True)
    
    sfm_start = time.time()
    scene = sparse_global_alignment(
        image_paths,
        pairs,
        cache_path,
        model,
        lr1=0.07, niter1=NITER1,
        lr2=0.014, niter2=NITER2,
        matching_conf_thr=0.01,  # Keep all matches, filter later
        shared_intrinsics=False,  # Better for head rotation
        subsample=8,
        device=device
    )
    sfm_time = time.time() - sfm_start
    print(f"  SfM done in {sfm_time:.1f}s")
    
    # 5. Save Results with Multiple Conf Thresholds
    print("\n[Step 5/5] Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get poses
    poses = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "camera_poses.npy"), poses.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "camera_focals.npy"), focals.astype(np.float32))
    print(f"  Poses: {poses.shape}")
    
    # Get points with confidence
    pts3d_list, _, confs_list = scene.get_dense_pts3d()
    all_pts, all_colors, all_confs = [], [], []
    
    for i, (pts, conf) in enumerate(zip(pts3d_list, confs_list)):
        all_pts.append(pts.reshape(-1, 3).cpu().numpy())
        all_confs.append(conf.reshape(-1).cpu().numpy())
        img_np = (imgs[i]["img"][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        all_colors.append(img_np.reshape(-1, 3))
    
    pts3d = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    confs = np.concatenate(all_confs, axis=0)
    colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    # Add conf as 4th column
    pts3d_with_conf = np.column_stack([pts3d, confs])
    
    print(f"  Total points: {len(pts3d):,}")
    
    # Save 3 versions
    for conf_thresh in [0.1, 0.5, 1.0]:
        ply_name = f"pointcloud_conf{conf_thresh:.1f}.ply"
        ply_path = os.path.join(OUTPUT_DIR, ply_name)
        n_pts = save_ply(pts3d_with_conf, colors, conf_thresh, ply_path)
        print(f"  Saved {ply_name}: {n_pts:,} points")
    
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Images (after blur filter): {len(image_paths)}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
