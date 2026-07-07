# MASt3R-SfM: The "Proper" Guide

> **Goal:** Turn any video into a high-quality 3D point cloud and camera poses ready for 3D Gaussian Splatting (3DGS) training.

This guide documents the robust "Full Graph" pipeline using `sparse_global_alignment` with high confidence thresholds, which yields the best results.

---

## 1. Prerequisites & Installation

**Hardware**: NVIDIA GPU (24GB+ VRAM recommended for "Large" model). S_01 / H100 recommended.

### Step 1: Clone Repository
```bash
git clone --recursive https://github.com/naver/mast3r
cd mast3r
```

### Step 2: Create Environment
Use Python 3.10 to avoid compatibility headaches.

```bash
conda create -n mast3r python=3.10 cmake=3.14.0 -y
conda activate mast3r

# Install PyTorch (adjust CUDA version if needed, e.g., 11.8 or 12.1)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
```

> **Note:** If you see a warning about "RoPE2D using slow pytorch version," it is safe to ignore for inference.

---

## 2. The Pipeline Script

Save this script as `run_mast3r_pipeline.py` in your `mast3r` directory (or parent). It handles:
1.  Loading images.
2.  Creating a **Complete** scene graph (pairs every image with every other image).
3.  Running **Sparse Global Alignment** (the core SfM).
4.  Exporting **CLEAN** Point Clouds (PLY) and Camera Poses (NPY).

```python
import sys
import os
import glob
import torch
import numpy as np
from plyfile import PlyData, PlyElement

# --- CONFIGURATION ---
MAST3R_PATH = '/teamspace/studios/this_studio/mast3r'  # Path to valid mast3r repo
IMAGE_DIR = './my_images'                             # Input images folder
OUTPUT_DIR = './output_mast3r'                        # Output folder
CONF_THRESHOLD = 1.0                                  # Min confidence (1.0 = strict/clean)
# ---------------------

sys.path.insert(0, MAST3R_PATH)

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from dust3r.utils.image import load_images

def main():
    # 1. Load Model
    print('Loading MASt3R model...')
    device = 'cuda'
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to(device)

    # 2. Load Images
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.png')))
    
    print(f'Found {len(image_paths)} images')
    # Load with fixed size (512 edge)
    imgs = load_images(image_paths, size=512, verbose=True)

    # 3. Create Pairs (Complete Graph)
    # 'complete' connects everyone (N^2 pairs). Best for < 50 images.
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f'Created {len(pairs)} pairs (complete graph)')

    # 4. Run Optimization
    cache_path = os.path.join(OUTPUT_DIR, 'cache')
    os.makedirs(cache_path, exist_ok=True)

    print('Running sparse_global_alignment...')
    scene = sparse_global_alignment(
        image_paths, 
        pairs, 
        cache_path, 
        model,
        lr1=0.07, niter1=500,    # Coarse alignment steps
        lr2=0.014, niter2=200,   # Fine alignment steps
        matching_conf_thr=CONF_THRESHOLD, 
        shared_intrinsics=True,  # Assume same camera for video
        subsample=8,
        device=device
    )

    # 5. Extract & Save Results
    print('Extracting results...')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Poses & Intrinsics
    poses = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    pps = scene.get_principal_points().detach().cpu().numpy()

    np.save(os.path.join(OUTPUT_DIR, 'camera_poses.npy'), poses.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, 'camera_focals.npy'), focals.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, 'camera_pps.npy'), pps.astype(np.float32))

    # Dense Point Cloud
    pts3d_list, colors_list, confs_list = scene.get_dense_pts3d()
    
    all_pts, all_colors, all_confs = [], [], []

    for i, (pts, conf) in enumerate(zip(pts3d_list, confs_list)):
        # pts is (H, W, 3)
        pts_np = pts.reshape(-1, 3).cpu().numpy()
        conf_np = conf.reshape(-1).cpu().numpy()
        
        # Get colors from original image tensor
        img_tensor = imgs[i]['img'] # (1, 3, H, W)
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy() # (H, W, 3)
        img_np = (img_np * 0.5 + 0.5) # De-normalize to 0..1
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

    print(f'Total points: {len(pts3d)}, Filtered (conf>={CONF_THRESHOLD}): {len(pts3d_filtered)}')

    # Write PLY
    vertices = np.zeros(len(pts3d_filtered), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertices['x'] = pts3d_filtered[:, 0]
    vertices['y'] = pts3d_filtered[:, 1]
    vertices['z'] = pts3d_filtered[:, 2]
    vertices['red'] = colors_filtered[:, 0]
    vertices['green'] = colors_filtered[:, 1]
    vertices['blue'] = colors_filtered[:, 2]

    ply_path = os.path.join(OUTPUT_DIR, 'pointcloud.ply')
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(ply_path)
    print(f'Saved pointcloud to {ply_path}')
    print('Done!')

if __name__ == '__main__':
    main()
```

---

## 3. Usage Guide

### Step 1: Prepare Your Video
Extract frames at a low frame rate (e.g., 1 FPS). This reduces redundancy and makes the "Complete" graph computationally feasible.

```bash
mkdir -p my_scene/images
ffmpeg -i video.mp4 -vf fps=1 my_scene/images/frame_%04d.jpg
```

### Step 2: Run the Pipeline
Edit `run_mast3r_pipeline.py` to point `IMAGE_DIR` to your `my_scene/images`.

```bash
python run_mast3r_pipeline.py
```

### Step 3: Check Output
You will get:
*   `camera_poses.npy`: (N, 4, 4) Camera-to-World poses.
*   `pointcloud.ply`: High-quality, colored point cloud.

---

## 4. Next Steps (3DGS Training)

To use this data for training vanilla 3D Gaussian Splatting:

1.  **Format for 3DGS**: You need to convert the NPY poses to COLMAP `sparse/0` format (`cameras.bin`, `images.bin`, `points3D.bin`).
    *   *Note: MASt3R poses are Cam-to-World. COLMAP uses World-to-Cam.*
    *   Invert poses: `w2c = np.linalg.inv(c2w)`
2.  **Train**:
    ```bash
    python train.py -s /path/to/my_scene
    ```

---

## 5. Bonus: Convert to COLMAP (for 3DGS)

Save this as `convert_to_colmap.py` to bridge MASt3R output to 3DGS.

```python
import os
import numpy as np
import argparse
import struct

def write_colmap_cameras(intrinsics, width, height, path):
    # ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # PINHOLE: fx, fy, cx, cy
    with open(path, 'w') as f:
        pass # Create file
    # Simple radial model often used, but PINHOLE is safer for basics
    # COLMAP format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS(fx, fy, cx, cy)
    # Binary format is complex, text format is easier.
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mast3r_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--width', type=int, default=1080) # Adjust!
    parser.add_argument('--height', type=int, default=1920) # Adjust!
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MASt3R data
    poses = np.load(os.path.join(args.mast3r_dir, 'camera_poses.npy')) # C2W
    focals = np.load(os.path.join(args.mast3r_dir, 'camera_focals.npy'))
    pps = np.load(os.path.join(args.mast3r_dir, 'camera_pps.npy'))
    
    # 1. Write cameras.txt
    # We assume shared intrinsics for video, but let's write per-image if needed or 1 shared.
    # Let's write one shared camera.
    fx = np.mean(focals)
    fy = fx
    cx = np.mean(pps[:, 0])
    cy = np.mean(pps[:, 1])
    
    with open(os.path.join(args.output_dir, 'cameras.txt'), 'w') as f:
        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
        f.write('# Camera list with one line of data per camera\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'1 PINHOLE {args.width} {args.height} {fx} {fy} {cx} {cy}\n')

    # 2. Write images.txt
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    with open(os.path.join(args.output_dir, 'images.txt'), 'w') as f:
        f.write('# Image list with two lines of data per image\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        
        for i in range(len(poses)):
            c2w = poses[i]
            # Convert to W2C
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            # Rotation matrix to quaternion
            # (Simple implementation or use scipy)
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_matrix(R)
            qx, qy, qz, qw = rot.as_quat() 
            
            # Name: frame_%04d.jpg ? Adjust matching your filenames
            name = f'frame_{i+1:04d}.jpg' 
            
            f.write(f'{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}\n')
            f.write('\n') # No 2D points needed for training usually if initialized with ply

    # 3. Write points3D.txt (Empty is fine if you initialize with PLY)
    with open(os.path.join(args.output_dir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point\n')

    print('Done! Copy your images to images/ folder next to sparse/')

if __name__ == '__main__':
    main()
```

