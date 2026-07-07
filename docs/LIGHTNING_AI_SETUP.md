# Lightning AI (H100) Studio Setup Guide

This guide documents the persistent environment configurations, compilation variables, and storage layouts used on Lightning AI H100 Studios to achieve high-throughput, error-free SfM and 3DGS training.

---

## 1. Directory Layout (Persistent Mount)

Because `/teamspace/studios/this_studio/` is a persistent storage mount that remains completely identical across studio stops and starts, we locate all repositories, virtual environments, and data inputs here. 

The scripts in `/scripts/` assume this exact layout and will run without path edits:

```
/teamspace/studios/this_studio/
├── py310_env/              # Python 3.10 virtual environment (survives restarts)
├── mast3r/                 # naver/mast3r repository
├── speedy_mast3r/          # ASU-ESIC-FAN-Lab/speedy_mast3r repository
├── 3rgs/                   # 3R-GS repository
├── CityGaussian/           # Linketic/CityGaussian repository
├── images/                 # Input frames extracted from video
└── output/                 # Results (poses, point clouds, PLY files)
```

---

## 2. Remote Access & File Synchronization

### SSH Access
Connect to your studio container using standard SSH routing (replace `s_YOUR_STUDIO_ID` with your active studio instance hash):
```bash
ssh s_YOUR_STUDIO_ID@ssh.lightning.ai
```

### Rclone Google Drive Sync
Transfer heavy raw video files or massive checkpoint `.ckpt` targets using `rclone`.
1. Upload your local `rclone.conf` key to the studio persistent root:
   ```bash
   scp ~/.config/rclone/rclone.conf s_YOUR_STUDIO_ID@ssh.lightning.ai:/teamspace/studios/this_studio/
   ```
2. Run rclone copy inside the studio:
   ```bash
   rclone --config=/teamspace/studios/this_studio/rclone.conf copy gdrive:MyWeddingClips ./raw_data
   ```

---

## 3. Persistent Virtual Environment Setup

Always create your virtual environments inside the persistent `/teamspace/` partition (not `/home/zeus/`) to prevent packages from being deleted when the studio goes to sleep.

```bash
# Initialize Python 3.10 env
conda create -p /teamspace/studios/this_studio/py310_env python=3.10 cmake=3.14.0 -y
conda activate /teamspace/studios/this_studio/py310_env

# Install PyTorch built against CUDA 12.1 or 11.8
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Activate venv by path directly
source /teamspace/studios/this_studio/py310_env/bin/activate
```

---

## 4. CUDA Kernel Compilation (RoPE)

MASt3R and Speedy MASt3R rely on a custom PyTorch C++ extension for Rotary Position Embeddings (RoPE). We must compile these extensions matching our GPU architecture (H100 uses Compute Capability `9.0`).

Run this in **both** repositories:

```bash
# Set H100 compilation target architecture
export TORCH_CUDA_ARCH_LIST="9.0"

# Compile in Speedy MASt3R
cd /teamspace/studios/this_studio/speedy_mast3r/dust3r/croco/models/curope
python setup.py build_ext --inplace

# Compile in Standard MASt3R
cd /teamspace/studios/this_studio/mast3r/dust3r/croco/models/curope
python setup.py build_ext --inplace
```

---

## 5. Rebuilding Gsplat After Studio Restarts

**Symptom:** Importing `torch` or starting training hangs indefinitely or crashes after a studio shutdown and manual restart.
**Cause:** Shared CUDA library linkages in dynamically compiled rasterizer environments get desynchronized across virtual machines.
**Solution:** Force rebuild the `gsplat` wheel under your current session environment using `--no-build-isolation`:

```bash
source /teamspace/studios/this_studio/py310_env/bin/activate
export MAX_JOBS=16

# Reinstall and re-compile gsplat
pip install --no-build-isolation \
  "git+https://github.com/nerfstudio-project/gsplat@ec3e715f5733df90d804843c7246e725582df10c"
```

---

## 6. One-Time H100 Speedup Patch

Apply the mixed precision and block size scaling directly using `sed` to avoid manual editing:

```bash
sed -i 's/block_size=2\*\*13/block_size=2**19, use_amp=True/' /teamspace/studios/this_studio/speedy_mast3r/mast3r/cloud_opt/sparse_ga.py
```
This forces Speedy MASt3R to process in massive blocks with Automatic Mixed Precision (AMP), decreasing processing times by up to **80%**.
