# Technical Challenges & Solutions

This document catalogs the exact engineering obstacles encountered while building, optimizing, and integrating various 3DGS-adjacent tools on Lightning AI (H100 GPUs), and the concrete solutions devised.

---

## Part 1: MASt3R & Speedy MASt3R Integration

### Challenge 1: Module Path Conflicts between Repositories
* **Symptom:** Importing modules from Speedy MASt3R after importing them from regular MASt3R (or vice versa) results in silent overrides, broken functions, or import errors. Both repositories use the identical namespace structures `mast3r` and `dust3r`.
* **Cause:** Python's module caching mechanism (`sys.modules`) retains the first imported version of any package. Inserting a new directory in `sys.path` does not force Python to reload files with identical package names.
* **Solution:** Explicitly purge all cached modules matching `mast3r` or `dust3r` in `sys.modules` prior to altering `sys.path` and importing the alternative package.
  ```python
  import sys
  for key in list(sys.modules.keys()):
      if "mast3r" in key or "dust3r" in key:
          del sys.modules[key]
  sys.path.insert(0, MAST3R_PATH)
  ```

### Challenge 2: Mismatched Cache & Prediction Output Keys
* **Symptom:** Using Speedy MASt3R outputs directly within MASt3R components crashes with key lookup errors.
* **Cause:** Speedy MASt3R uses custom output keys optimized for speed, whereas the standard MASt3R optimizer expects distinct keys. Specifically, `pred1` utilizes the key `"pts3d"`, but `pred2` uses the key `"pts3d_in_other_view"` (NOT `"pts3d"`).
* **Solution:** Map the keys carefully when caching:
  ```python
  res1 = output["pred1"]["pts3d"][idx]
  res2 = output["pred2"]["pts3d_in_other_view"][idx] # Correct mapping!
  ```

### Challenge 3: Inconsistent Return Signature in `fast_reciprocal_NNs`
* **Symptom:** Calling `fast_reciprocal_NNs` occasionally throws `ValueError: too many values to unpack` or crashes.
* **Cause:** Depending on the execution pathway and environment, `fast_reciprocal_NNs` returns 2 values in some environments, and 3 values (including confidence weight outputs) in others.
* **Solution:** Capture the return tuple as a generic list or tuple, check its length dynamically, and default the confidence scores to a tensor of ones if missing:
  ```python
  result = matches
  xy1, xy2 = result[0], result[1]
  confs_m = result[2] if len(result) > 2 else torch.ones(len(xy1))
  ```

### Challenge 4: PyTorch Negative Stride Conversion Crash
* **Symptom:** Converting cached NumPy coordinates to PyTorch tensors throws `ValueError: At least one stride in the given numpy array is negative`.
* **Cause:** Index slicing or array flipping in NumPy creates arrays with negative strides. PyTorch tensors can only be initialized from contiguous, positive-stride memory blocks.
* **Solution:** Force contiguous layout using `np.ascontiguousarray` before instantiating the tensor.
  ```python
  xy1_t = torch.tensor(np.ascontiguousarray(xy1))
  ```

### Challenge 5: Cache Folder Structure Misalignment
* **Symptom:** MASt3R's `sparse_global_alignment` fails to read cached inference pairs and re-runs heavy inference, losing any speed benefits.
* **Cause:** MASt3R's optimizer expects a highly strict nested cache directory layout.
* **Solution:** Replicate the exact folder structure when dumping the speedy inference cache:
  ```
  cache/
  ├── forward/{idx1}/{idx2}.pth              # Contains (pts3d, conf, pts3d_other, conf_other)
  └── corres_conf=desc_conf_subsample=8/
      └── {idx1}-{idx2}.pth                  # Contains (score, (xy1, xy2, confs))
  ```

---

## Part 2: GPU Performance & Processing Quality (H100)

### Challenge 6: Severe Processing Bottlenecks on H100 GPUs
* **Symptom:** MASt3R pair inference is extremely slow (~5 iterations/sec), failing to leverage the H100's high throughput.
* **Cause:** The default `block_size` in standard MASt3R is set to `2**13` (8,192 elements), which causes up to 32 loop cycles per pair. This severely underutilizes massive tensor cores.
* **Solution:** Patch `sparse_ga.py` to scale the internal `block_size` up to `2**19` (524,288 elements) and activate Mixed Precision training (`use_amp=True`). This condenses 32 loops into 1 loop, generating a **~5x speedup**.
  ```python
  # Location: speedy_mast3r/mast3r/cloud_opt/sparse_ga.py line 604
  # Replace:
  opt = dict(device=device, dist='dot', block_size=2**13)
  # With:
  opt = dict(device=device, dist='dot', block_size=2**19, use_amp=True)
  ```

### Challenge 7: Blurry Frames Corrupting Structure-from-Motion Loss
* **Symptom:** High optimization loss (e.g. > 0.70) during global alignment, leading to floating noise and geometry separation.
* **Cause:** Blurry frames (e.g. from camera panning, movement, wedding scenes) inject high reprojection errors into the SfM solver.
* **Solution:** Run a Laplacian variance filter before committing images to the SfM step. Drop frames falling below a strict threshold (variance < 30). This filters out ~10-20% of problematic frames, dropping loss down to **~0.08** (excellent quality).
  ```python
  def compute_blur_score(img_path):
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      return cv2.Laplacian(img, cv2.CV_64F).var() if img is not exists else 0
  ```

### Challenge 8: Splat Separation / Face Cracking on Rotating Subjects
* **Symptom:** Faces or primary subjects develop visible cracks or geometry splits when the subject rotates.
* **Cause:** Assuming `shared_intrinsics=True` for videos helps stability but breaks when the subject rotates or zooms, causing perspective distortions that a single intrinsic matrix cannot model.
* **Solution:** Set `shared_intrinsics=False` in `sparse_global_alignment` when capturing head/face rotations to estimate independent focal lengths per frame.

---

## Part 3: CityGaussian Export & Viewer Compatibility

### Challenge 9: Pink or Inverted Colors in Splat Viewers (SuperSplat, Polycam)
* **Symptom:** The exported `.ply` file displays with pink, inverted, or completely wrong color hues in standard web viewers.
* **Cause:** Spherical Harmonics (SH) coefficients represent angular color variations. Vanilla 3DGS expects these coefficients grouped color-by-color (all R coefficients, then G, then B). CityGaussian stores them grouped by SH degree component `(N, SH_coeffs, 3)`.
* **Solution:** Transpose the dimensions `(1, 2)` of the SH tensors before flattening them to group them by color channel.
  ```python
  # Correct transpose for Spherical Harmonics representation:
  # shs_dc is (N, 1, 3) -> transpose -> (N, 3, 1) -> flatten -> (N, 3)
  f_dc = shs_dc.transpose(1, 2).flatten(start_dim=1).float().numpy()
  # shs_rest is (N, 15, 3) -> transpose -> (N, 3, 15) -> flatten -> (N, 45)
  f_rest = shs_rest.transpose(1, 2).flatten(start_dim=1).float().numpy()
  ```

### Challenge 10: Invisible or Microscopic Exported Scenes
* **Symptom:** Loading the exported CityGaussian PLY shows an empty screen. Zooming out extremely far reveals a microscopic dot.
* **Cause:** CityGaussian checkpoints save scale factors in log-space. Exporters that attempt to rescale spatial variables mathematically by multiplying or dividing the scale attributes shrink them down to microscopic scales.
* **Solution:** Do not apply manual scaling factors to scale fields during export. Keep original scale attributes, but subtract the centroid from coordinates to center the camera navigation at `(0, 0, 0)`.
  ```python
  center = xyz.mean(dim=0)
  xyz = xyz - center # Center coordinates but leave scale log-factors alone!
  ```

---

## Part 4: Depth-Anything-3 (DA3) Streaming

### Challenge 11: `NoneType flatten` Crashes during Streaming
* **Symptom:** Processing video frames through DA3 crashes with an error referring to flattening a `NoneType` object.
* **Cause:** The configuration parameter `chunk_size` is larger than the total number of frames in the video, causing the chunking alignment algorithm to return null groupings.
* **Solution:** Dynamically reduce `chunk_size` or force it to be smaller than the video length (e.g. 10-30 for short clips).

### Challenge 12: No PLY Point Cloud Generated
* **Symptom:** Depth maps render correctly, but no merged `.ply` point cloud is created in the output directory.
* **Cause:** DA3 only runs global point cloud fusion when crossing boundaries between sequential chunks. If the entire video fits inside a single chunk, the merger trigger is bypassed.
* **Solution:** Force partition processing by lowering `chunk_size` or ensuring chunking is explicitly triggered.

### Challenge 13: Low Detail / Ghosting on Close-up Subjects
* **Symptom:** Face details, hands, or wedding rings are blurred, smeared, or exhibit phantom geometry (ghosting).
* **Cause:** The default processing resolution is set to **504p** internally to preserve VRAM, losing high-frequency depth details.
* **Solution:** Patch the resolution processing keyword in `da3_streaming.py` line 273 up to **768** or **1024** (if > 24GB VRAM is available).
  ```python
  predictions = self.model.inference(images, ref_view_strategy=ref_view_strategy, process_res=768)
  ```

---

## Part 5: 3R-GS & Poses

### Challenge 14: 3R-GS Portrait Stride Decoder Stalling (The Final Wall)
* **Symptom:** Pose optimization during 3R-GS training diverges immediately, corrupting point clouds on vertical/portrait captures.
* **Cause:** 3R-GS hardcodes a stride division of `512` internally to decode flattened pixel coordinates of correspondences. Portrait-oriented video captures use a compressed landscape width of `288` pixels internally in MASt3R, breaking decoding math.
* **Solution:** Skip 3R-GS joint pose optimization. Initialize `gsplat` MCMC baseline training directly using the custom-formatted MASt3R-SfM camera poses and dense point cloud.

### Challenge 15: MASt3R Cam-to-World Poses vs COLMAP World-to-Cam format
* **Symptom:** Importing MASt3R poses directly into standard 3DGS environments puts the camera positions inside objects or shoots them off to infinity.
* **Cause:** MASt3R generates Camera-to-World (C2W) transformation poses. COLMAP (`sparse/0/images.bin`) and standard 3DGS packages expect World-to-Camera (W2C) transformations.
* **Solution:** Apply a matrix inversion to the 4x4 coordinate matrices during the COLMAP generation step.
  ```python
  c2w = poses[i]
  w2c = np.linalg.inv(c2w) # Invert to World-to-Camera!
  ```
