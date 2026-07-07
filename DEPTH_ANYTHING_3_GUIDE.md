# DA3-Streaming: Quick Guide

> **Goal:** Turn video into point cloud + camera poses using Depth-Anything 3.

---

## 1. Setup (Lightning Studio)

```bash
cd /home/zeus/content
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3

export PATH=/home/zeus/content/py310_env/bin:$PATH
pip install -r requirements.txt
pip install -e .
pip install pypose numba

# Download weights
bash da3_streaming/scripts/download_weights.sh

# Create symlink for weights
cd da3_streaming && ln -sf ../weights weights
```

---

## 2. Extract Frames

```bash
mkdir -p /home/zeus/content/my_video/images
ffmpeg -i video.mp4 -vf "fps=10" /home/zeus/content/my_video/images/frame_%05d.jpg
```

---

## 3. Config Options

Create `config.yaml`:
```yaml
Weights:
  DA3: "./weights/model.safetensors"
  DA3_CONFIG: "./weights/config.json"
  SALAD: "./weights/dino_salad.ckpt"

Model:
  chunk_size: 30       # Smaller = more robust for faces/indoor
  overlap: 15          # 50% overlap recommended
  loop_enable: True    # Global refinement
  align_lib: "torch"
  align_method: "sim3"
  save_depth_conf_result: True

  Pointcloud_Save:
    sample_ratio: 0.05       # 5% = moderate density
    conf_threshold_coef: 0.5 # Lower = more points (0.8 = strict)
```

### Key Settings:
| Need | chunk_size | overlap | conf |
|------|------------|---------|------|
| Indoor/Face | 10-30 | 50% | 0.5 |
| Outdoor/Street | 60-120 | 50% | 0.7-0.8 |

---

## 4. Resolution (IMPORTANT!)

Default is **504p**. To change, patch `da3_streaming.py` line ~273:
```python
# Change from:
predictions = self.model.inference(images, ref_view_strategy=ref_view_strategy)
# To:
predictions = self.model.inference(images, ref_view_strategy=ref_view_strategy, process_res=768)
```

Options: 504 (default), 768, 1024 (needs more VRAM)

---

## 5. Run

```bash
export PATH=/home/zeus/content/py310_env/bin:$PATH
cd /home/zeus/content/Depth-Anything-3/da3_streaming

python da3_streaming.py \
  --image_dir /path/to/images \
  --output_dir /path/to/output \
  --config /path/to/config.yaml
```

---

## 6. Output

```
output/
├── camera_poses.txt      # 4x4 matrices per frame
├── intrinsic.txt         # fx, fy, cx, cy per frame  
├── pcd/combined_pcd.ply  # Merged point cloud
└── results_output/       # Per-frame depth + confidence
```

---

## 7. Common Issues

| Error | Fix |
|-------|-----|
| `NoneType flatten` | chunk_size > frames → use smaller chunks |
| No PLY output | Single chunk processed → force chunking |
| Low face detail | Increase process_res (768/1024) |
| Ghost geometry | Use MASt3R-SfM instead (better SfM) |
