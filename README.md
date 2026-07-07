# 3DGS Moments — Capturing People, Not Just Places

> A comprehensive research repository documenting the journey, technical challenges, and optimized scripts for capturing real human moments (e.g. couples at weddings) as walk-through 3D Gaussian Splats.

> ⏸️ **Status: Shelved / Paused.** This repository is preserved as a permanent notes directory and learning log. It tracks what worked, what failed, and the optimizations that saved hours of debugging.

---

## ⭐ Major Discoveries & Engineering Wins

If you are trying to reproduce high-quality 3DGS or SfM models of dynamic human subjects, these are the key discoveries and fixes found throughout this research. Detailed walkthroughs for each of these are documented in **[docs/CHALLENGES.md](docs/CHALLENGES.md)**.

| Focus Area | Discovery & Optimization | Practical Impact | Pipeline Stage |
| :--- | :--- | :--- | :--- |
| **Performance** | **H100 Tensor Core Fix:** Expand `block_size` from `2**13` to `2**19` and enable Mixed Precision (`use_amp=True`) in MASt3R's `sparse_ga.py`. | **~5× Speedup** on H100 GPUs (reducing loop count from 32 to 1 per pair). | [02_speedy_mast3r](docs/pipelines/02_speedy_mast3r.md) |
| **Quality** | **Blur Pre-Filtering:** Apply Laplacian variance threshold check (drop frames with variance < 30) before committing to SfM. | **Loss dropped from 0.70 to 0.08** (visible noise artifacts eliminated). | [01_mast3r_sfm](docs/pipelines/01_mast3r_sfm.md) |
| **Color Rendering** | **SH Transpose Fix:** Transpose Spherical Harmonics dimensions `(1, 2)` before flattening to export. | Solves **pink, wrong, or inverted colors** in web viewers (SuperSplat, Polycam). | [export_script](scripts/export/export_citygaussian_ply.py) |
| **Scale / Nav** | **Origin Centering over Scale Modification:** Center coordinates at origin but do not scale logarithmic scale factors on checkpoint export. | Solves **invisible, microscopic, or clipped scenes** on import. | [03_citygaussian](docs/pipelines/03_citygaussian.md) |
| **Distortion** | **`shared_intrinsics=False`:** Disable intrinsic camera parameters sharing on global alignment. | Prevents **cracks and holes in faces** when the subject rotates. | [01_mast3r_sfm](docs/pipelines/01_mast3r_sfm.md) |
| **Structure** | **Pose Matrix Inversion:** Invert Camera-to-World poses to World-to-Camera for COLMAP sparse folders. | Fixes broken coordinate initializations for vanilla 3DGS. | [01_mast3r_sfm](docs/pipelines/01_mast3r_sfm.md) |

---

## 📽️ The Core Objective

Most 3D Gaussian Splatting configurations capture environments: streets, statues, or empty rooms. This research was focused on capturing **human memories**: couples celebrating, vertical mobile captures of people, and panning close-ups. 

Because dynamic subjects, hair detail, facial expressions, and panning cameras introduce severe noise into traditional Structure-from-Motion (SfM) pipelines, a specialized hybrid workflow is required.

---

## 🛠️ The Pipeline

```
                       [ Input Video Capture ]
                                  │ (ffmpeg 1-4 fps)
                                  ▼
                     [ Blur Filtering (Laplacian) ]
                                  │ (Drop frames < 30 var)
                                  ▼
                      [ Structure-from-Motion ]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼ (Primary)                                       ▼ (Alternative)
   [ MASt3R-SfM ]                                     [ Depth-Anything-3 ]
   - Speedy Hybrid Caching                            - 768p Process Res
   - Independ. Intrinsics                             - Short Chunk Size
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
                     [ Initial Poses & Point Cloud ]
                                  │ (Pose Matrix Inversion)
                                  ▼
                    [ 3D Gaussian Splatting Trainer ]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼ (Works)                                         ▼ (Blocked)
   [ CityGaussian Joint Opt ]                         [ 3R-GS Joint Opt ]
   - MCMC-3DGS Config                                 - Fails on Portrait Stride (288)
   - 30k Steps                                        - Hardcoded % 512 Stride Error
         │
         ▼
   [ Final PLY Export Script ]
   - Center Coords
   - Fix SH Ordering
         │
         ▼
 [ Standard Web Viewers (SuperSplat) ]
```

---

## 📦 What I Tried

Our evaluations spanned 6 different model frameworks:

* **[MASt3R-SfM](docs/pipelines/01_mast3r_sfm.md) (naver):** Video to camera poses + point cloud. (Status: **✅ Works beautifully**)
* **[Speedy MASt3R](docs/pipelines/02_speedy_mast3r.md) (ASU):** Accelerating pairwise neural network inference. (Status: **✅ Works in hybrid cache mode**)
* **[CityGaussian](docs/pipelines/03_citygaussian.md) (Linketic):** Joint pose refinement and fast splat training. (Status: **✅ Works with custom exporter**)
* **Vanilla 3DGS (gsplat MCMC):** Solid, reliable training benchmark. (Status: **✅ Works with ~1° pose error**)
* **[Depth-Anything-3](docs/pipelines/05_depth_anything_3.md) (ByteDance):** Depth-based SfM alternative. (Status: **⚠️ Partial** - exhibits ghost geometry on humans)
* **[3R-GS](docs/pipelines/04_3rgs.md):** Dynamic joint optimization. (Status: **❌ Blocked** - fails on portrait videos due to hardcoded % 512 match stride)

---

## 📊 Results Summary

Quality results tracked by SfM global alignment loss (**< 0.1 = Excellent, < 0.5 = Good, > 0.7 = Artifacts**):

| Subject Clip | Clip Duration | Frames Tracked | Alignment Loss (Stage 1/2) | Visual Reconstruction Quality |
| :--- | :---: | :---: | :---: | :--- |
| **rakhi sitting** | 34s | 38 / 68 | **0.08 / 0.60** | ⭐⭐⭐ (Excellent, sharp contours) |
| **rakhi** | 22s | 42 / 43 | **0.08 / 0.53** | ⭐⭐⭐ (Excellent facial contours) |
| **yugtest** | 43s | 74 / 86 | 0.45 / 0.79 | ⭐⭐ (Good background, minor halo) |
| **ishu** | 18s | 64 / 71 | 0.74 / 0.95 | ⭐ (Heavy noise, floating artifacts) |

---

## 📩 Trained .ply / .splat Files

The trained models represent **personal memories and captures** of real people, making them difficult to host in a public space. 

If you would like to see, run, or test the resulting `.ply` or `.splat` files of these scenes, please send a DM on Instagram:
👉 **[@mevikasrao](https://instagram.com/mevikasrao)**

---

## 💻 Repository Structure

```
3dgs-moments/
├── docs/
│   ├── CHALLENGES.md             # In-depth logs of solved issues and fixes
│   ├── LIGHTNING_AI_SETUP.md     # Persistent studio setup, compilation, and gsplat help
│   ├── REFERENCES.md             # Links to original academic papers & repos
│   └── pipelines/
│       ├── 01_mast3r_sfm.md      # Canonical MASt3R setup and guide
│       ├── 02_speedy_mast3r.md   # Speedy integration and timing benchmark
│       ├── 03_citygaussian.md    # CityGaussian joint pose training instructions
│       ├── 04_3rgs.md            # Details of the 3R-GS portrait stride block
│       └── 05_depth_anything_3.md# Depth-Anything-3 Streaming pipeline settings
└── scripts/
    ├── mast3r_sfm/               # Custom run scripts (progress bars, speedy hybrids, etc.)
    └── export/
        └── export_citygaussian_ply.py # Fixes SH order and centers coordinate system
```

---

## 💡 About lightning.ai Environments

All scripts are configured to utilize path conventions native to **Lightning AI H100 Studios**. 

Because `/teamspace/studios/this_studio/` is a persistent cluster mount that resolves identically for any user on any instance, **the scripts run as-is out-of-the-box inside Lightning AI**. You only need to clone the external repositories to this directory and activate the custom environment. Full setup guidelines are available in **[docs/LIGHTNING_AI_SETUP.md](docs/LIGHTNING_AI_SETUP.md)**.
