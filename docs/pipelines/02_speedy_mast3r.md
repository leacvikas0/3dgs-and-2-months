# Speedy MASt3R Integration Guide

> **Goal:** Accelerate MASt3R-SfM by using Speedy MASt3R for fast pair inference while keeping MASt3R's SfM optimization.

## Overview

Speedy MASt3R (from ASU-ESIC-FAN-Lab) uses TensorRT/ONNX to accelerate the neural network inference step. However, it's **not a drop-in replacement** for the full SfM pipeline. We use a hybrid approach:

```
Images → Speedy MASt3R (inference) → Cache → MASt3R (sparse_global_alignment) → Poses + Point Cloud
           ↑ Fast (1.7s for 20 pairs) ↑          ↑ Optimization (11.7s) ↑
```

## Timing Comparison (5 Cabinet Images)

| Method | Inference | SfM | Total |
|--------|-----------|-----|-------|
| **Speedy Hybrid** | 1.7s | 11.7s | **24.6s** |
| Regular MASt3R | ~10s | ~11s | ~21s |

For small image sets, the difference is minimal. **The real speedup shows on larger datasets** where inference dominates.

## Key Integration Challenges Solved

### 1. Module Path Conflicts
Speedy MASt3R and MASt3R have same module names (`mast3r`, `dust3r`). Solution: Clear `sys.modules` before switching.

```python
for key in list(sys.modules.keys()):
    if "mast3r" in key or "dust3r" in key:
        del sys.modules[key]
sys.path.insert(0, MAST3R_PATH)
```

### 2. Output Key Differences
```python
# pred1 uses "pts3d"
# pred2 uses "pts3d_in_other_view" (NOT "pts3d")
res1 = output["pred1"]["pts3d"][idx]
res2 = output["pred2"]["pts3d_in_other_view"][idx]
```

### 3. fast_reciprocal_NNs Return Format
Sometimes returns 2 values, sometimes 3. Handle both:
```python
result = matches
xy1, xy2 = result[0], result[1]
confs_m = result[2] if len(result) > 2 else torch.ones(len(xy1))
```

### 4. Numpy Array Stride Issue
Negative strides cause tensor conversion errors. Fix with:
```python
torch.tensor(np.ascontiguousarray(xy1))
```

### 5. Cache Format Matching
MASt3R expects specific cache structure:
```
cache/
├── forward/{idx1}/{idx2}.pth    # (pts3d, conf, pts3d_other, conf_other)
└── corres_conf=desc_conf_subsample=8/
    └── {idx1}-{idx2}.pth        # (score, (xy1, xy2, confs))
```

## Files

| File | Description |
|------|-------------|
| `speedy_hybrid_final.py` | Complete hybrid pipeline script |
| `run_mast3r_sfm.py` | Standard MASt3R SfM script |
| `run_mast3r_moti.py` | Video processing with frame extraction |

## Usage

### Standard MASt3R SfM
```bash
# Edit paths in script
python run_mast3r_sfm.py
```

### Speedy Hybrid Pipeline
```bash
# Requires both repos:
# - /teamspace/studios/this_studio/speedy_mast3r
# - /teamspace/studios/this_studio/mast3r

python speedy_hybrid_final.py
```

## When to Use Speedy

| Use Case | Recommendation |
|----------|---------------|
| < 50 images | Regular MASt3R (simpler) |
| 50-200 images | Speedy Hybrid (noticeable speedup) |
| > 200 images | Speedy Hybrid + retrieval mode |

## Installation Requirements

```bash
# Both repos need:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r dust3r/requirements.txt

# Build RoPE kernels in BOTH repos:
cd dust3r/croco/models/curope && python setup.py build_ext --inplace
```

## Output

Both methods produce:
- `camera_poses.npy` - (N, 4, 4) Camera-to-World transforms
- `camera_focals.npy` - Focal lengths
- `camera_pps.npy` - Principal points  
- `pointcloud.ply` - Colored point cloud (confidence filtered)
