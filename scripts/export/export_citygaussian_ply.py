import torch
import numpy as np
from plyfile import PlyData, PlyElement
import sys
import argparse
import os

# Script to export CityGaussian checkpoints to viewer-compatible PLY format
# Key feature: Correctly transposes SH coefficients to match vanilla 3DGS format
# Usage: python export_citygaussian_ply.py --ckpt path/to/ckpt --output path/to/output.ply

def export_citygaussian_ply(ckpt_path, output_path):
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        # Map to CPU to avoid CUDA requirements for simple export
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Handle different checkpoint structures (Lightning vs vanilla)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # Key mapping for CityGaussian (based on GaussianEditor/yzslab)
    # These usage keys are specific to CityGaussian's internal model
    keys_map = {
        "xyz": "gaussian_model.gaussians.means",
        "shs_dc": "gaussian_model.gaussians.shs_dc",
        "shs_rest": "gaussian_model.gaussians.shs_rest",
        "scale": "gaussian_model.gaussians.scales",
        "rot": "gaussian_model.gaussians.rotations",
        "opacity": "gaussian_model.gaussians.opacities"
    }

    try:
        xyz = sd[keys_map["xyz"]]
        shs_dc = sd[keys_map["shs_dc"]]     # (N, 1, 3)
        shs_rest = sd[keys_map["shs_rest"]]  # (N, 15, 3)
        scale = sd[keys_map["scale"]]       # (N, 3) - log space
        rot = sd[keys_map["rot"]]           # (N, 4) - quaternion
        opacity = sd[keys_map["opacity"]]   # (N, 1) - logit space
    except KeyError as e:
        print(f"KeyError: {e}. Keys not found in checkpoint. Checkpoint keys: {list(sd.keys())[:5]}...")
        return

    n = xyz.shape[0]
    print(f"Loaded {n} gaussians")

    # Center at origin for viewer convenience
    # CityGaussian scenes are often huge (map coords), centering helps navigation
    center = xyz.mean(dim=0)
    xyz = xyz - center

    # --- CRITICAL FIX FOR SH FORMAT ---
    # Vanilla 3DGS save_ply expects flattened SH coefficients in distinct order:
    # [R_sh0..15, G_sh0..15, B_sh0..15]
    # CityGaussian stores as (N, SH_coeffs, 3).
    # We must TRANSPOSE(1, 2) before flattening to group by color channel.
    # Data is (N, 1, 3)
    f_dc = shs_dc.transpose(1, 2).flatten(start_dim=1).float().numpy()     # (N, 3)
    
    # Data is (N, 15, 3) -> transpose(1,2) -> (N, 3, 15) -> flatten -> (N, 45)
    f_rest = shs_rest.transpose(1, 2).flatten(start_dim=1).float().numpy()  # (N, 45)

    # Build PLY dtype
    dtype = [("x","f4"),("y","f4"),("z","f4"),
             ("nx","f4"),("ny","f4"),("nz","f4"),
             ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4")]
    
    # 45 rest coefficients for SH degree 3
    for i in range(45): 
        dtype.append((f"f_rest_{i}","f4"))
    
    dtype += [("opacity","f4"),
              ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
              ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]

    print("Building PLY array...")
    elements = np.empty(n, dtype=dtype)
    elements["x"] = xyz[:,0].float().numpy()
    elements["y"] = xyz[:,1].float().numpy()
    elements["z"] = xyz[:,2].float().numpy()
    elements["nx"] = elements["ny"] = elements["nz"] = 0
    
    elements["f_dc_0"] = f_dc[:,0]
    elements["f_dc_1"] = f_dc[:,1]
    elements["f_dc_2"] = f_dc[:,2]
    
    # Copy rest coefficients
    for i in range(45): 
        elements[f"f_rest_{i}"] = f_rest[:,i]
    
    elements["opacity"] = opacity[:,0].float().numpy()
    elements["scale_0"] = scale[:,0].float().numpy()
    elements["scale_1"] = scale[:,1].float().numpy()
    elements["scale_2"] = scale[:,2].float().numpy()
    elements["rot_0"] = rot[:,0].float().numpy()
    elements["rot_1"] = rot[:,1].float().numpy()
    elements["rot_2"] = rot[:,2].float().numpy()
    elements["rot_3"] = rot[:,3].float().numpy()

    print(f"Writing to {output_path}...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CityGaussian checkpoint to viewer-compatible PLY")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--output", required=True, help="Path to output .ply file")
    args = parser.parse_args()
    
    export_citygaussian_ply(args.ckpt, args.output)
