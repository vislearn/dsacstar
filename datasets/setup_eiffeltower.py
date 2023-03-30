import os
import cv2
import torch
import pycolmap
import numpy as np
from pathlib import Path


# The Eiffel Tower underwater dataset is available at:
# https://www.seanoe.org/data/00810/92226/


def download_eiffeltower(output_dir: Path):
    dataset_files = {
        '2015': '98240.zip',
        '2016': '98289.zip',
        '2018': '98314.zip',
        '2020': '98356.zip',
        'global': '98357.zip'
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'Directory {output_dir} already exists, please delete it before setting up the dataset.')
    for dataset, dataset_file in dataset_files.items():
        print(f'Download Eiffel Tower {dataset} dataset.')
        os.system(f'wget https://www.seanoe.org/data/00810/92226/data/{dataset_file} -P {output_dir}')
        os.system(f'unzip {output_dir / dataset_file} -d {output_dir}')
        os.system(f'rm {output_dir / dataset_file}')


if __name__ == '__main__':

    nn_subsampling = 8  # sub sampling of our CNN architecture, for size of the initalization targets
    target_height = 480  # rescale images
    out_h = int(np.ceil(target_height / nn_subsampling))

    # Set up the dataset directory
    dataset_dir = Path('EiffelTower')

    # Download the dataset
    download_eiffeltower(dataset_dir)

    # Set up train and test sets, images of 2015 visit are test and the rest are train
    split_dirs = {
        '2015': dataset_dir / 'test',
        '2016': dataset_dir / 'train',
        '2018': dataset_dir / 'train',
        '2020': dataset_dir / 'train'
    }
    for split_dir in set(split_dirs.values()):
        for data_dir in ['rgb', 'init', 'poses', 'calibration']:
            (split_dir / data_dir).mkdir(parents=True)

    print('Load Eiffel Tower model.')
    colmap_model = pycolmap.Reconstruction(dataset_dir / 'global' / 'sfm')

    print('Load 3D points to facilitate initialization set up.')
    points3D = np.zeros((max(colmap_model.point3D_ids()) + 1, 3), dtype=np.float64)
    for point3D_id, point3D in colmap_model.points3D.items():
        points3D[point3D_id] = point3D.xyz

    for i, image in enumerate(colmap_model.images.values()):
        print(f'[{i+1:05d}/{colmap_model.num_images()}] Process {image.name}.')

        # Get image year visit
        year = image.name[:4]

        # Set up output directory
        split_dir = split_dirs[year]  # image names start with the year

        # Get camera information
        camera = colmap_model.cameras[image.camera_id]

        # Compute output target width
        target_width = int(np.ceil(camera.width * target_height / camera.height))
        out_w = int(np.ceil(target_width / nn_subsampling))

        # Get camera matrix
        camera_matrix = camera.calibration_matrix()

        # Get camera distortion coefficients
        k1, k2 = camera.params[3:]
        dist_coeffs = np.array([k1, k2, 0, 0])

        # Compute optimal camera matrix with centered principal point
        # given the model's camera matrix and distortion coefficients
        new_camera_matrix, (roix, roiy, roiw, roih) = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            imageSize=(camera.width, camera.height),
            alpha=0,
            newImgSize=(target_width, target_height),
            centerPrincipalPoint=True
        )

        # Get focal length of new camera matrix
        focal_length = new_camera_matrix[0, 0]

        # Save calibration
        calib_path = (split_dir / 'calibration' / image.name).with_suffix('.txt')
        calib_path.write_text(f'{focal_length}')

        # Read image
        im = cv2.imread(str(dataset_dir / year / 'images' / image.name))

        # Undistort, resize and center principal point of image
        im = cv2.undistort(im, camera_matrix, dist_coeffs, newCameraMatrix=new_camera_matrix)
        im = im[roiy:roiy + roih, roix:roix + roiw]

        # Save image
        cv2.imwrite(str(split_dir / 'rgb' / image.name), im)

        # Load world-to-camera pose
        t = image.tvec.reshape(3, 1)
        R = image.rotation_matrix()

        # Setup init file for scene coordinates regression
        point3D_ids = [point2D.point3D_id for point2D in image.get_valid_points2D()]  # ids of all visible 3D points
        w_P = points3D[point3D_ids]  # 3D points in the world frame
        c_P = R @ w_P.T + t  # 3D points in the camera frame
        c_p = new_camera_matrix @ c_P  # project 3D points on image plane in homogeneous coordinates
        c_p = c_p[:2] / c_p[2]  # 2D points on the image plane
        c_p = np.int32(c_p / nn_subsampling)  # 2D points scaled to fit the CNN output
        c_p[0] = c_p[0].clip(0, out_w - 1)
        c_p[1] = c_p[1].clip(0, out_h - 1)
        args = np.argsort(c_P[2])[::-1]  # sort pixels by depth
        out_tensor = torch.zeros((3, out_h, out_w))
        for (x, y), P in zip(c_p.T[args], torch.tensor(w_P[args])):  # valid 3D coordinate is the closest one
            out_tensor[:, y, x] = P

        # Save init file
        torch.save(out_tensor, (split_dir / 'init' / image.name).with_suffix('.dat'))

        # Inverse pose to set it to camera-to-world
        t = -R.T @ t
        R = R.T

        # Save pose
        pose_path = (split_dir / 'poses' / image.name).with_suffix('.txt')
        pose_path.write_text(
            f'{R[0, 0]} {R[0, 1]} {R[0, 2]} {t[0, 0]}\n'
            f'{R[1, 0]} {R[1, 1]} {R[1, 2]} {t[1, 0]}\n'
            f'{R[2, 0]} {R[2, 1]} {R[2, 2]} {t[2, 0]}\n'
            f'0.0 0.0 0.0 1.0\n'
        )

    print('Processing of the Eiffel Tower dataset completed.')
