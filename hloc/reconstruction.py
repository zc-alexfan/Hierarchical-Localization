import argparse
import shutil
from typing import Optional, List, Dict, Any
import multiprocessing
from pathlib import Path
import pycolmap
import os.path as osp
import numpy as np

from . import logger
from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features, import_matches, estimation_and_geometric_verification,
    OutputCapture, parse_option_args)


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()

def load_intrin_to_database(output_db_path, intrin_prior_path, colmap_cfgs=None):
    assert osp.exists(intrin_prior_path)
    single_camera = False
    if colmap_cfgs is not None and "ImageReader_single_camera" in colmap_cfgs:
        if colmap_cfgs["ImageReader_single_camera"]:
            single_camera = True

    db = COLMAPDatabase.connect(output_db_path)
    # Check num of camera:
    rows = db.execute("SELECT camera_id FROM cameras")
    camera_ids = [id[0] for id in rows]
    if len(camera_ids) == 1:
        assert osp.isfile(intrin_prior_path) and single_camera, f"single_camera is switched, however given a intrin directory"

        row = db.execute(f"SELECT width, height FROM cameras WHERE camera_id = {camera_ids[0]}")
        w, h = next(row)
        db.execute(f"DELETE FROM cameras WHERE camera_id = {camera_ids[0]}")

        K = np.loadtxt(intrin_prior_path) # 3*3
        fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]

        db.add_camera(1, w, h, np.array((fx, fy, cx, cy)), camera_id=camera_ids[0])

    else:
        # Load image name, camera id from images:
        for image_name, camera_id in db.execute("SELECT name, camera_id FROM images"):
            # Delete camera:
            row = db.execute(f"SELECT width, height FROM cameras WHERE camera_id = {camera_id}")
            w, h = next(row)
            db.execute(f"DELETE FROM cameras WHERE camera_id = {camera_id}")

            # Then add new camera:
            img_base_name = osp.splitext(osp.basename(image_name))[0]
            assert osp.isdir(intrin_prior_path), f"Provided intrinsics path is not a directory! You need to switch single_camera for providing only one intrinsic file "

            intrin_prior_file_path = osp.join(intrin_prior_path, img_base_name+'.txt')
            K = np.loadtxt(intrin_prior_file_path)
            fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]

            db.add_camera(1, w, h, np.array((fx, fy, cx, cy)), camera_id=camera_id)

    db.commit()
    db.close()

def import_images(image_dir: Path,
                  database_path: Path,
                  camera_mode: pycolmap.CameraMode,
                  image_list: Optional[List[str]] = None,
                  options: Optional[Dict[str, Any]] = None):
    logger.info('Importing images into the database...')
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    with pycolmap.ostream():
        pycolmap.import_images(database_path, image_dir, camera_mode,
                               image_list=image_list or [],
                               options=options)


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(sfm_dir: Path,
                       database_path: Path,
                       image_dir: Path,
                       verbose: bool = False,
                       colmap_configs: Optional[Dict[str, Any]] = None,
                       ) -> pycolmap.Reconstruction:
    models_path = sfm_dir / 'models'
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info('Running 3D reconstruction...')
    if colmap_configs is None:
        colmap_configs = {}
    # options = {'num_threads': min(multiprocessing.cpu_count(), 16), **options}
    mapper_options = pycolmap.IncrementalMapperOptions(ba_global_use_pba=colmap_configs['use_pba'], ba_refine_focal_length=not colmap_configs['no_refine_intrinsics'], ba_refine_extra_params=not colmap_configs['no_refine_intrinsics'], num_threads=min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16))
    with OutputCapture(verbose):
        with pycolmap.ostream():
            logger.info(f"use: {min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16)} cpus")
            logger.info(mapper_options.summary())
            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path, options=mapper_options)

    if len(reconstructions) == 0:
        logger.error('Could not reconstruct any model!')
        return None
    logger.info(f'Reconstructed {len(reconstructions)} model(s).')

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(f'Largest model is #{largest_index} '
                f'with {largest_num_images} images.')

    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(
            str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]


def main(sfm_dir: Path,
         image_dir: Path,
         intrinsic_f: Path,
         pairs: Path,
         features: Path,
         matches: Path,
         camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
         verbose: bool = False,
         skip_geometric_verification: bool = False,
         min_match_score: Optional[float] = None,
         image_list: Optional[List[str]] = None,
         image_options: Optional[Dict[str, Any]] = None,
         mapper_options: Optional[Dict[str, Any]] = None,
         ) -> pycolmap.Reconstruction:

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    colmap_configs = {'ImageReader_single_camera': True, 
                      'min_model_size': 3, 
                      'filter_max_reproj_error': 4, 
                      'no_refine_intrinsics': True, 
                      'ImageReader_camera_mode': 'single_camera', 
                      'use_pba': False, 
                      'n_threads': 16, 
                      'reregistration': {'abs_pose_max_error': 12, 'abs_pose_min_num_inliers': 30, 'abs_pose_min_inlier_ratio': 0.25, 'filter_max_reproj_error': 5}, 
                      'colmap_mapper_cfgs': None}
    load_intrin_to_database(database, intrinsic_f, colmap_configs)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)
    reconstruction = run_reconstruction(
        sfm_dir, database, image_dir, verbose, colmap_configs)
    if reconstruction is not None:
        logger.info(f'Reconstruction statistics:\n{reconstruction.summary()}'
                    + f'\n\tnum_input_images = {len(image_ids)}')
    return reconstruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--camera_mode', type=str, default="AUTO",
                        choices=list(pycolmap.CameraMode.__members__.keys()))
    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--image_options', nargs='+', default=[],
                        help='List of key=value from {}'.format(
                            pycolmap.ImageReaderOptions().todict()))
    parser.add_argument('--mapper_options', nargs='+', default=[],
                        help='List of key=value from {}'.format(
                            pycolmap.IncrementalMapperOptions().todict()))
    args = parser.parse_args().__dict__

    image_options = parse_option_args(
        args.pop("image_options"), pycolmap.ImageReaderOptions())
    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions())

    main(**args, image_options=image_options, mapper_options=mapper_options)
