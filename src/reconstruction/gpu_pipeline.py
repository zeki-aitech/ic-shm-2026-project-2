"""
GPU-accelerated re-triangulation pipeline (requires pycolmap-cuda12).

The contest sparse cloud has only ~86k points because it keeps the original
SfM tracks. With known (fixed) camera poses we can re-detect features, re-match
them on the GPU and triangulate a much denser sparse cloud — COLMAP's
``point_triangulator`` workflow:

1. Stage all 400 images (labeled + unlabeled) into one flat directory.
2. SIFT feature extraction on GPU, with the known SIMPLE_RADIAL intrinsics
   fixed (single shared camera).
3. Sequential feature matching on GPU (UAV frames form an ordered sequence).
4. Build a reference reconstruction holding the known poses (registered
   frames, no 3D points).
5. ``pycolmap.triangulate_points`` triangulates a new point cloud while
   keeping poses and intrinsics fixed.

Run from the repo root:  python3 -m src.reconstruction.gpu_pipeline
"""
import os
import shutil
import time

import pycolmap

from src.reconstruction.pycolmap_reconstructor import load_contest_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.getenv("CONTEST_DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "Contest Dataset"))
COLMAP_DIR = os.path.join(DATASET_DIR, "camera_parameters")
WORK_DIR = os.path.join(PROJECT_ROOT, "outputs", "gpu_pipeline")


def stage_images(dataset_dir: str, image_dir: str) -> int:
    """Symlink labeled + unlabeled images into one flat directory."""
    os.makedirs(image_dir, exist_ok=True)
    count = 0
    for sub in ["images", "unlabeled_Images"]:
        src_dir = os.path.join(dataset_dir, sub)
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(".png"):
                continue
            dst = os.path.join(image_dir, fname)
            if not os.path.exists(dst):
                os.symlink(os.path.join(src_dir, fname), dst)
            count += 1
    return count


def extract_and_match(database_path: str, image_dir: str,
                      camera: pycolmap.Camera, overlap: int = 20,
                      max_num_features: int = 16384,
                      peak_threshold: float = 0.004,
                      matcher: str = "sequential") -> None:
    """GPU SIFT extraction (fixed intrinsics, single camera) + sequential matching."""
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera.model_name
    reader_options.camera_params = ",".join(str(p) for p in camera.params)

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = max_num_features
    # Default peak threshold (~0.0067) yields only ~4.5k keypoints per image on
    # this dataset; lower it to detect more features for a denser cloud.
    extraction_options.sift.peak_threshold = peak_threshold

    t0 = time.time()
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cuda,
    )
    print(f"[gpu] Feature extraction done in {time.time() - t0:.1f}s")

    t0 = time.time()
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path=database_path, device=pycolmap.Device.cuda)
    else:
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = overlap
        pycolmap.match_sequential(
            database_path=database_path,
            pairing_options=pairing_options,
            device=pycolmap.Device.cuda,
        )
    print(f"[gpu] {matcher} matching done in {time.time() - t0:.1f}s")


def build_reference_model(database_path: str,
                          contest_model: pycolmap.Reconstruction) -> pycolmap.Reconstruction:
    """
    Build a reconstruction whose image ids match the feature database and whose
    poses come from the contest model (matched by file name). Frames are
    registered so triangulate_points treats every pose as known and fixed.
    """
    poses_by_name = {img.name: img.cam_from_world() for img in contest_model.images.values()}

    db = pycolmap.Database.open(database_path)
    db_images = db.read_all_images()
    db_camera = db.read_all_cameras()[0]
    db.close()

    ref = pycolmap.Reconstruction()
    ref.add_camera_with_trivial_rig(db_camera)

    for db_image in db_images:
        pose = poses_by_name.get(db_image.name)
        if pose is None:
            continue
        image = pycolmap.Image()
        image.image_id = db_image.image_id
        image.camera_id = db_camera.camera_id
        image.name = db_image.name
        ref.add_image_with_trivial_frame(image)
        ref.frames[db_image.image_id].rig_from_world = pose
        ref.register_frame(db_image.image_id)

    print(f"[gpu] Reference model: {ref.num_images()} images with known poses "
          f"({ref.num_reg_frames()} registered)")
    return ref


def run(overlap: int = 20, max_num_features: int = 16384,
        peak_threshold: float = 0.002,
        matcher: str = "sequential") -> pycolmap.Reconstruction:
    os.makedirs(WORK_DIR, exist_ok=True)
    image_dir = os.path.join(WORK_DIR, "images")
    database_path = os.path.join(WORK_DIR, "database.db")
    output_dir = os.path.join(WORK_DIR, "triangulated")

    if not pycolmap.has_cuda:
        raise RuntimeError("This pipeline requires the CUDA build (pip install pycolmap-cuda12)")

    n_images = stage_images(DATASET_DIR, image_dir)
    print(f"[gpu] Staged {n_images} images -> {image_dir}")

    contest_model = load_contest_model(COLMAP_DIR)
    camera = contest_model.cameras[next(iter(contest_model.cameras))]

    if os.path.exists(database_path):
        os.remove(database_path)  # rerun from scratch to keep the db consistent
    extract_and_match(database_path, image_dir, camera, overlap=overlap,
                      max_num_features=max_num_features, peak_threshold=peak_threshold,
                      matcher=matcher)

    ref = build_reference_model(database_path, contest_model)

    t0 = time.time()
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    rec = pycolmap.triangulate_points(
        reconstruction=ref,
        database_path=database_path,
        image_path=image_dir,
        output_path=output_dir,
        refine_intrinsics=False,
    )
    print(f"[gpu] Triangulation done in {time.time() - t0:.1f}s")

    rec.export_PLY(os.path.join(WORK_DIR, "dense_sparse_cloud.ply"))
    print(f"[gpu] Result: {rec.num_points3D()} 3D points "
          f"(contest model had 86,336 track ids)")
    print(f"[gpu] Mean track length: {rec.compute_mean_track_length():.1f}, "
          f"mean reprojection error: {rec.compute_mean_reprojection_error():.2f}px")
    print(f"[gpu] Model written to {output_dir}, PLY to {WORK_DIR}/dense_sparse_cloud.ply")
    return rec


if __name__ == "__main__":
    run()
