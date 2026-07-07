import sys, os, glob, torch, numpy as np, time

SPEEDY_PATH = "/teamspace/studios/this_studio/speedy_mast3r"
MAST3R_PATH = "/teamspace/studios/this_studio/mast3r"
IMAGE_DIR = "/teamspace/studios/this_studio/test_data/Cabinet"
OUTPUT_DIR = "/teamspace/studios/this_studio/cabinet_speedy_hybrid"
CONF_THRESHOLD = 1.0

def run_speedy_inference():
    """Run speedy inference and cache results"""
    sys.path.insert(0, SPEEDY_PATH)
    
    start = time.time()
    device = "cuda"
    
    print("Loading Speedy MASt3R model...")
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.utils.image import load_images
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs as dust3r_make_pairs
    
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    model.eval()
    print(f"Model loaded in {time.time()-start:.1f}s")
    
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    print(f"Found {len(image_paths)} images")
    imgs = load_images(image_paths, size=512, verbose=True)
    
    pairs = dust3r_make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} pairs")
    
    print("Running Speedy inference...")
    inference_start = time.time()
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    inference_time = time.time() - inference_start
    print(f"Inference done in {inference_time:.1f}s")
    
    # Cache results
    cache_path = os.path.join(OUTPUT_DIR, "cache")
    os.makedirs(cache_path, exist_ok=True)
    
    print("Caching results...")
    n_pairs = len(output["pred1"]["pts3d"])
    
    for idx in range(n_pairs):
        res1 = output["pred1"]["pts3d"][idx]
        res2 = output["pred2"]["pts3d_in_other_view"][idx]
        conf1 = output["pred1"]["conf"][idx]
        conf2 = output["pred2"]["conf"][idx]
        desc1 = output["pred1"]["desc"][idx]
        desc2 = output["pred2"]["desc"][idx]
        
        view1 = output["view1"]
        view2 = output["view2"]
        idx1 = view1["idx"][idx].item() if torch.is_tensor(view1["idx"][idx]) else view1["idx"][idx]
        idx2 = view2["idx"][idx].item() if torch.is_tensor(view2["idx"][idx]) else view2["idx"][idx]
        
        matches = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device)
        
        fwd_dir = os.path.join(cache_path, f"forward/{idx1}")
        os.makedirs(fwd_dir, exist_ok=True)
        torch.save((res1.cpu(), conf1.cpu(), res2.cpu(), conf2.cpu()), 
                   os.path.join(fwd_dir, f"{idx2}.pth"))
        
        corres_dir = os.path.join(cache_path, "corres_conf=desc_conf_subsample=8")
        os.makedirs(corres_dir, exist_ok=True)
        result = matches
        xy1, xy2 = result[0], result[1]
        confs_m = result[2] if len(result) > 2 else torch.ones(len(xy1))
        
        if xy1 is not None and len(xy1) > 0:
            score = confs_m.mean().item() if torch.is_tensor(confs_m) else float(np.mean(confs_m))
            xy1_t = xy1.cpu() if torch.is_tensor(xy1) else torch.tensor(np.ascontiguousarray(xy1))
            xy2_t = xy2.cpu() if torch.is_tensor(xy2) else torch.tensor(np.ascontiguousarray(xy2))
            confs_t = confs_m.cpu() if torch.is_tensor(confs_m) else torch.tensor(confs_m)
        else:
            xy1_t, xy2_t, confs_t = torch.zeros(0,2), torch.zeros(0,2), torch.zeros(0)
            score = 0.0
        torch.save((score, (xy1_t, xy2_t, confs_t)), os.path.join(corres_dir, f"{idx1}-{idx2}.pth"))
    
    print(f"Cached {n_pairs} pairs")
    return image_paths, imgs, cache_path, model, inference_time

def run_sfm(image_paths, imgs, cache_path, model):
    """Run SfM using MASt3R"""
    # Clear speedy imports and use mast3r
    for key in list(sys.modules.keys()):
        if "mast3r" in key or "dust3r" in key:
            del sys.modules[key]
    sys.path.insert(0, MAST3R_PATH)
    
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    
    # Reload images for mast3r format
    imgs = load_images(image_paths, size=512, verbose=False)
    pairs_mast3r = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    
    print("Running sparse_global_alignment...")
    sfm_start = time.time()
    scene = sparse_global_alignment(
        image_paths, pairs_mast3r, cache_path, model,
        lr1=0.07, niter1=500, lr2=0.014, niter2=200,
        matching_conf_thr=CONF_THRESHOLD, shared_intrinsics=True, subsample=8, device="cuda"
    )
    sfm_time = time.time() - sfm_start
    print(f"SfM done in {sfm_time:.1f}s")
    
    return scene, imgs, sfm_time

def save_results(scene, imgs):
    from plyfile import PlyData, PlyElement
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    poses = scene.get_im_poses().detach().cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "camera_poses.npy"), poses.astype(np.float32))
    print(f"Saved poses: {poses.shape}")
    
    pts3d_list, _, confs_list = scene.get_dense_pts3d()
    all_pts, all_colors, all_confs = [], [], []
    for i, (pts, conf) in enumerate(zip(pts3d_list, confs_list)):
        all_pts.append(pts.reshape(-1, 3).cpu().numpy())
        all_confs.append(conf.reshape(-1).cpu().numpy())
        all_colors.append((imgs[i]["img"][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).reshape(-1, 3))
    
    pts3d, colors, confs = np.concatenate(all_pts), np.concatenate(all_colors), np.concatenate(all_confs)
    mask = confs >= CONF_THRESHOLD
    pts3d_f, colors_f = pts3d[mask], (np.clip(colors[mask], 0, 1) * 255).astype(np.uint8)
    print(f"Points: {len(pts3d)} -> {len(pts3d_f)} filtered")
    
    vertices = np.zeros(len(pts3d_f), dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
    vertices["x"], vertices["y"], vertices["z"] = pts3d_f[:,0], pts3d_f[:,1], pts3d_f[:,2]
    vertices["red"], vertices["green"], vertices["blue"] = colors_f[:,0], colors_f[:,1], colors_f[:,2]
    
    ply_path = os.path.join(OUTPUT_DIR, "pointcloud.ply")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(ply_path)
    print(f"Saved {ply_path}")

if __name__ == "__main__":
    total_start = time.time()
    image_paths, imgs, cache_path, model, inference_time = run_speedy_inference()
    scene, imgs, sfm_time = run_sfm(image_paths, imgs, cache_path, model)
    save_results(scene, imgs)
    print(f"\n=== TIMING ===")
    print(f"Speedy Inference: {inference_time:.1f}s")
    print(f"SfM Optimization: {sfm_time:.1f}s")
    print(f"TOTAL: {time.time()-total_start:.1f}s")
