# 3DGS Moments — Capturing People, Not Just Places

> A comprehensive research repository documenting the journey, technical challenges, and optimized scripts for capturing real human moments (e.g. couples at weddings) as walk-through 3D Gaussian Splats.

> ⏸️ **Status: Shelved / Paused.** This repository is preserved as a permanent notes directory and learning log. It tracks what worked, what failed, and the optimizations that saved hours of debugging.

---

## ⭐ Major Discoveries & Engineering Wins

If you are trying to reproduce high-quality 3DGS or SfM models of dynamic human subjects, these are the key discoveries and fixes found throughout this research. Detailed walkthroughs for each of these are documented in **[CHALLENGES.md](CHALLENGES.md)**.

| Focus Area | Discovery & Optimization | Practical Impact | Pipeline Stage Guide |
| :--- | :--- | :--- | :--- |
| **Performance** | **H100 Tensor Core Fix:** Expand `block_size` from `2**13` to `2**19` and enable Mixed Precision (`use_amp=True`) in MASt3R's `sparse_ga.py`. | **~5× Speedup** on H100 GPUs (reducing loop count from 32 to 1 per pair). | [SPEEDY_MAST3R_GUIDE.md](SPEEDY_MAST3R_GUIDE.md) |
| **Quality** | **Blur Pre-Filtering:** Apply Laplacian variance threshold check (drop frames with variance < 30) before committing to SfM. | **Loss dropped from 0.70 to 0.08** (visible noise artifacts eliminated). | [MAST3R_SFM_GUIDE.md](MAST3R_SFM_GUIDE.md) |
| **Color Rendering** | **SH Transpose Fix:** Transpose Spherical Harmonics dimensions `(1, 2)` before flattening to export. | Solves **pink, wrong, or inverted colors** in web viewers (SuperSplat, Polycam). | [export_citygaussian_ply.py](export_citygaussian_ply.py) |
| **Scale / Nav** | **Origin Centering over Scale Modification:** Center coordinates at origin but do not scale logarithmic scale factors on checkpoint export. | Solves **invisible, microscopic, or clipped scenes** on import. | [CITYGAUSSIAN_GUIDE.md](CITYGAUSSIAN_GUIDE.md) |
| **Distortion** | **`shared_intrinsics=False`:** Disable intrinsic camera parameters sharing on global alignment. | Prevents **cracks and holes in faces** when the subject rotates. | [MAST3R_SFM_GUIDE.md](MAST3R_SFM_GUIDE.md) |
| **Structure** | **Pose Matrix Inversion:** Invert Camera-to-World poses to World-to-Camera for COLMAP sparse folders. | Fixes broken coordinate initializations for vanilla 3DGS. | [MAST3R_SFM_GUIDE.md](MAST3R_SFM_GUIDE.md) |

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

* **[MASt3R-SfM](MAST3R_SFM_GUIDE.md) (naver):** Video to camera poses + point cloud. (Status: **✅ Works beautifully**)
* **[Speedy MASt3R](SPEEDY_MAST3R_GUIDE.md) (ASU):** Accelerating pairwise neural network inference. (Status: **✅ Works in hybrid cache mode**)
* **[CityGaussian](CITYGAUSSIAN_GUIDE.md) (Linketic):** Joint pose refinement and fast splat training. (Status: **✅ Works with custom exporter**)
* **Vanilla 3DGS (gsplat MCMC):** Solid, reliable training benchmark. (Status: **✅ Works with ~1° pose error**)
* **[Depth-Anything-3](DEPTH_ANYTHING_3_GUIDE.md) (ByteDance):** Depth-based SfM alternative. (Status: **⚠️ Partial** - exhibits ghost geometry on humans)
* **[3R-GS](3R_GS_GUIDE.md):** Dynamic joint optimization. (Status: **❌ Blocked** - fails on portrait videos due to hardcoded % 512 match stride)

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

Every pipeline guide and custom runner script has been brought directly to the root of the repository for immediate visibility and rapid access:

```
3dgs-moments/
├── README.md                 # This narrative and summary
├── CHALLENGES.md             # Detailed engineering log of all solved errors & fixes
├── LIGHTNING_AI_SETUP.md     # Setup notes, GPU compilations, and dynamic environment guides
├── REFERENCES.md             # Links to official publications and repositories
│
├── MAST3R_SFM_GUIDE.md       # Canonical MASt3R extraction & pose running guide
├── SPEEDY_MAST3R_GUIDE.md    # Speedy integration & timing benchmarks
├── CITYGAUSSIAN_GUIDE.md     # Joint pose training & viewer exporting guides
├── 3R_GS_GUIDE.md            # Details on the 3R-GS portrait % 512 stride block
├── DEPTH_ANYTHING_3_GUIDE.md # Alternative depth alignment & processing guide
│
├── run_quality.py            # Flagship pre-filtering & multi-conf SfM script
├── speedy_hybrid_final.py    # Dual-repo fast inference & caching engine
├── export_citygaussian_ply.py# Solves SH order color bugs and centers viewports
└── run_*.py                  # Alternative helper runners and progress bar scripts
```

---

## 💡 About lightning.ai Environments

All scripts are configured to utilize path conventions native to **Lightning AI H100 Studios**. 

Because `/teamspace/studios/this_studio/` is a persistent cluster mount that resolves identically for any user on any instance, **the scripts run as-is out-of-the-box inside Lightning AI**. You only need to clone the external repositories to this directory and activate the custom environment. Full setup guidelines are available in **[LIGHTNING_AI_SETUP.md](LIGHTNING_AI_SETUP.md)**.
