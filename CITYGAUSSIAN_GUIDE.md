# CityGaussian Setup & Usage

CityGaussian refines the camera poses (Joint Pose Optimization) and trains high-quality 3D Gaussians.

## 1. Installation

**Requirements:**
- Python 3.10
- PyTorch 2.3+
- CUDA 11.8 or 12.1

**Install Dependencies:**
```bash
# Basic deps
pip install lightning==2.3.3 viser open3d kornia tensorboard wandb tqdm plyfile

# Install submodules (YZslab fork of gsplat is critical)
# If cloning fresh:
git clone --recursive https://github.com/Linketic/CityGaussian.git
cd CityGaussian

# Install gsplat (specific fork required for joint opt)
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/gsplat  # Ensure this builds successfully
```

## 2. Training (Joint Pose Optimization)

We use the MCMC-3DGS config for joint pose optimization. This works best with MASt3R-SfM input.

**Command:**
```bash
# Set environment variables for efficient training
export MAX_JOBS=16  # Adjust based on CPU cores

# Run training
# --data.path: Path to your MASt3R-SfM output (containing sparse/0/)
python main.py fit \
    --config configs/colmap_pose_opt_mcmc.yaml \
    --data.path /path/to/mast3r_output \
    --optimizer.lr.spatial_lr_scale 1.0
```

*Note: Training typically takes ~30k steps.*

## 3. Exporting to Web Viewers
**CRITICAL:** CityGaussian's default PLY does not work in web viewers (SuperSplat, Polycam) due to:
1. Huge coordinate system (map coordinates)
2. SH coefficient ordering mismatch

Use the provided script to export correctly:

```bash
# Run from the CityGaussian directory
python ../scripts/export_citygaussian_ply.py \
    --ckpt outputs/your_experiment/checkpoints/epoch=xxx-step=30000.ckpt \
    --output final_scene.ply
```

This script will:
- Center the scene at (0,0,0)
- Fix the Spherical Harmonics format
- Produce a `.ply` file ready for SuperSplat.
