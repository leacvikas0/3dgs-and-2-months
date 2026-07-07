import sys, os, glob, torch, numpy as np, time
MAST3R_PATH = "/teamspace/studios/this_studio/mast3r"
IMAGE_DIR = "/teamspace/studios/this_studio/test_data/moti_frames"
OUTPUT_DIR = "/teamspace/studios/this_studio/moti_output"
CONF_THRESHOLD = 1.0

sys.path.insert(0, MAST3R_PATH)
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from plyfile import PlyData, PlyElement

def main():
    start = time.time()
    print("Loading MASt3R...")
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").cuda()
    print(f"Model loaded in {time.time()-start:.1f}s")

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    print(f"Found {len(image_paths)} images")
    imgs = load_images(image_paths, size=512, verbose=True)

    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} pairs (complete graph)")

    os.makedirs(os.path.join(OUTPUT_DIR, "cache"), exist_ok=True)
    print("Running sparse_global_alignment (niter1=650, niter2=260)...")
    sfm_start = time.time()
    scene = sparse_global_alignment(
        image_paths, pairs, os.path.join(OUTPUT_DIR, "cache"), model,
        lr1=0.07, niter1=650, lr2=0.014, niter2=260,
        matching_conf_thr=CONF_THRESHOLD, shared_intrinsics=True, subsample=8, device="cuda"
    )
    print(f"SfM done in {time.time()-sfm_start:.1f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    poses = scene.get_im_poses().detach().cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "camera_poses.npy"), poses.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "camera_focals.npy"), scene.get_focals().detach().cpu().numpy().astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "camera_pps.npy"), scene.get_principal_points().detach().cpu().numpy().astype(np.float32))
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
    print(f"Points: {len(pts3d)} -> {len(pts3d_f)} filtered (conf>={CONF_THRESHOLD})")

    vertices = np.zeros(len(pts3d_f), dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
    vertices["x"], vertices["y"], vertices["z"] = pts3d_f[:,0], pts3d_f[:,1], pts3d_f[:,2]
    vertices["red"], vertices["green"], vertices["blue"] = colors_f[:,0], colors_f[:,1], colors_f[:,2]
    ply_path = os.path.join(OUTPUT_DIR, "pointcloud.ply")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(ply_path)
    print(f"Saved {ply_path}")
    print(f"Total time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
