# 3R-GS Pipeline & Dataset Format

> **Goal:** Run 3R-GS training with joint pose optimization using MASt3R-SfM camera poses and dense correspondences.
> 
> **Status:** ❌ **Blocked / Failed** 
> 
> **The Real Failure:** 3R-GS completely fails when converting a standard COLMAP sparse directory (`sparse/0/` containing `.bin` or `.txt` formats) into the custom binary arrays (`.npy` files) required for its joint pose-optimizing training process. Because this conversion logic is buggy and undocumented, standard SfM outputs cannot be translated into a usable 3R-GS input dataset.

---

## 1. The Core Blocker: COLMAP to 3R-GS Conversion Crash

While 3R-GS claims to optimize camera poses jointly using global correspondence matching, it does not read COLMAP outputs directly during optimization. Instead, it relies on a translation layer (such as the undocumented `create_colmap_sparse.py` script) to map standard camera parameters, image databases, and 3D points into a proprietary structure.

During this conversion process:
1. **Binary Format Mismatch:** The code fails to parse the standard binary format of COLMAP files (`cameras.bin`, `images.bin`, `points3D.bin`), throwing key and struct errors during unpack cycles.
2. **Missing Intrinsics/Pose Mappings:** Even when converting binary files to text (`.txt`) formats, the conversion script crashes on parameter mapping, as it fails to properly map variable distortion parameters or non-shared intrinsics into the structured `.npy` parameters.
3. **Array Generation Failure:** Because the parser crashes, the critical tracking matrices (`ei.npy`, `ej.npy`, `corr_i.npy`, `corr_j.npy`, `corr_weight.npy`, etc.) are never populated, halting training before it can even initialize.

---

## 2. 3R-GS Expected Dataset Structure (Unachievable)

To train, 3R-GS expects a pre-converted folder layout. Due to the conversion failure, generating the files inside `/mast3r/` is currently impossible from normal COLMAP outputs:

```
dataset/
├── images/               # JPEG images
├── mast3r/               # ❌ CANNOT GENERATE THESE DUE TO CONVERSION FAILURE
│   ├── camera_poses.npy       # (N, 4, 4) cam-to-world
│   ├── camera_intrinsics.npy  # (N, 3, 3) K matrices
│   ├── depthmaps.npy          # (N, H, W) depth
│   ├── pointcloud.ply         # Initial pointcloud (e.g. from MASt3R filtering)
│   ├── ei.npy, ej.npy         # Pair indices (int32)
│   ├── corr_i.npy, corr_j.npy  # Flat pixel indices (int32)
│   ├── corr_weight.npy        # Confidence weights (float32)
│   ├── corr_mask.npy          # Valid mask (bool)
│   └── corr_batch_idx.npy     # All zeros (int64)
├── sparse/0/             # cameras.bin, images.bin, points3D.bin (Standard COLMAP format)
├── images_train.txt      # Image basenames (no extension! e.g., frame_00001)
├── images_test.txt       # Normally empty or holds eval test images
├── pose_gt_train.npy     # Copy of camera_poses.npy for training evaluation
└── pose_gt_test.npy      # Empty (0, 4, 4) array
```

---

## 3. Correspondence Encoding Issues (The Secondary Obstacle)

Even if the dataset is manually bootstrapped without the conversion script, the correspondence matching matrix breaks on portrait captures due to a hardcoded stride in the index decoder.

The cache folder contains files named like `{hash1}-{hash2}.pth` inside:
`cache_path/corres_conf=desc_conf_subsample=8/`

Each file contains:
```python
score, (xy1, xy2, confs) = torch.load(corres_file)
# xy1, xy2: (N, 2) pixel coordinates in (x, y) format
# confs: (N,) confidence weights
```

### Stride Encoding Bug
3R-GS expects a flat index for correspondences, encoded as:
`flat_index = y * MAST3R_WIDTH + x`

However:
- **Portrait images (H > W):** MASt3R scales and processes images at **288 × 512** internally. Hence, `MAST3R_WIDTH = 288`.
- **Landscape images (W > H):** MASt3R processes at **512 × 288** internally. Hence, `MAST3R_WIDTH = 512`.

**The Bug:** 3R-GS hardcodes a stride of `512` during index decoding (line 395 in its internal `mast3r.py` code):
`y = flat_index // 512` and `x = flat_index % 512`.

Because of this hardcoded stride:
* **Landscape videos** can be successfully parsed.
* **Portrait videos** (most typical mobile phone captures of wedding couples, vertical panning, etc.) completely break. The flat index calculated with a stride of `288` gets decoded incorrectly using a stride of `512`, corrupting the coordinates, throwing off the matching, and leading to catastrophic pose optimization divergence or crashes.

---

## 4. Workarounds and Next Steps

1. **Use Vanilla 3DGS Instead:** Avoid pose optimization. Initialize `gsplat` with MASt3R's filtered point cloud and poses, but skip joint optimization. This works beautifully and yields excellent quality.
2. **Use CityGaussian instead of 3R-GS:** CityGaussian successfully optimizes camera poses jointly (Joint Pose Optimization) using standard COLMAP directories without requiring custom proprietary `.npy` translation arrays. Follow the [CITYGAUSSIAN_GUIDE.md](CITYGAUSSIAN_GUIDE.md) to run it.
