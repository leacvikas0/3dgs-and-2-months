#!/usr/bin/env python3
"""
Speedy MASt3R SfM Pipeline with Full Progress Display
Uses Speedy MASt3R's sparse_global_alignment which:
- Uses flash-attn for faster inference
- Shows pair-by-pair progress with tqdm
- Shows global optimization progress
"""
import sys
import os
import glob
import torch
import numpy as np
import time

# ============ CONFIGURATION ============
SPEEDY_PATH = "/teamspace/studios/this_studio/speedy_mast3r"
IMAGE_DIR = "/teamspace/studios/this_studio/images"
OUTPUT_DIR = "/teamspace/studios/this_studio/output"
CONF_THRESHOLD = 1.0
NITER1 = 500
NITER2 = 200
# ========================================

sys.path.insert(0, SPEEDY_PATH)

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images


def main():
    print("=" * 60)
    print("SPEEDY MAST3R SFM PIPELINE (with flash-attn)")
    print("=" * 60)
    
    total_start = time.time()
    device = "cuda"
    
    # 1. Load Model
    print("\n[Step 1/4] Loading Speedy MASt3R model...")
    model_start = time.time()
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)
    print(f"  Model loaded in {time.time() - model_start:.1f}s")
    
    # 2. Load Images
    print("\n[Step 2/4] Loading images...")
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    if not image_paths:
        print(f"ERROR: No images found in {IMAGE_DIR}")
        return
    
    print(f"  Found {len(image_paths)} images")
    imgs = load_images(image_paths, size=512, verbose=True)
    
    # 3. Create Pairs
    print("\n[Step 3/4] Creating image pairs...")
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    n_pairs = len(pairs)
    print(f"  Created {n_pairs} pairs (complete graph)")
    
    # 4. Run Global Alignment
    cache_path = os.path.join(OUTPUT_DIR, "cache")
    os.makedirs(cache_path, exist_ok=True)
    
    print("\n[Step 4/4] Running sparse_global_alignment...")
    print(f"  Cache: {cache_path}")
    print(f"  niter1={NITER1}, niter2={NITER2}")
    print(f"  conf_threshold={CONF_THRESHOLD}")
    print("-" * 60)
    
    sfm_start = time.time()
    scene = sparse_global_alignment(
        image_paths,
        pairs,
        cache_path,
        model,
        lr1=0.07, niter1=NITER1,
        lr2=0.014, niter2=NITER2,
        matching_conf_thr=CONF_THRESHOLD,
        shared_intrinsics=True,
        subsample=8,
        device=device
    )
    sfm_time = time.time() - sfm_start
    print("-" * 60)
    print(f"  SfM completed in {sfm_time:.1f}s")
    
    # 5. Extract & Save Results
    print("\n[Saving Results]")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Poses
    poses = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    pps = scene.get_principal_points().detach().cpu().numpy()
    
    np.save(os.path.join(OUTPUT_DIR, "camera_poses.npy"), poses.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "camera_focals.npy"), focals.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "camera_pps.npy"), pps.astype(np.float32))
    print(f"  Poses: {poses.shape}")
    
    # Point Cloud
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
    
    # Filter
    mask = confs >= CONF_THRESHOLD
    pts3d_f = pts3d[mask]
    colors_f = (np.clip(colors[mask], 0, 1) * 255).astype(np.uint8)
    
    print(f"  Points: {len(pts3d):,} -> {len(pts3d_f):,} (conf>={CONF_THRESHOLD})")
    
    # Write PLY
    from plyfile import PlyData, PlyElement
    vertices = np.zeros(len(pts3d_f), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    vertices["x"] = pts3d_f[:, 0]
    vertices["y"] = pts3d_f[:, 1]
    vertices["z"] = pts3d_f[:, 2]
    vertices["red"] = colors_f[:, 0]
    vertices["green"] = colors_f[:, 1]
    vertices["blue"] = colors_f[:, 2]
    
    ply_path = os.path.join(OUTPUT_DIR, "pointcloud.ply")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(ply_path)
    print(f"  Saved: {ply_path}")
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Images: {len(image_paths)}")
    print(f"  Pairs: {n_pairs}")
    print(f"  Points: {len(pts3d_f):,}")
    print(f"  SfM time: {sfm_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
