"""
Convert EgoBody Kinect dataset to HumanData NPZ format.

Usage:
    python data/EgoBody_Kinect/create_npz.py \
        --egobody_dir D:/Dev/Dataset/EgoBody \
        --smplx_model_path common/utils/human_model_files \
        --output_dir D:/Dev/Dataset/preprocessed_datasets
"""

import os
import os.path as osp
import numpy as np
import pickle
import csv
import argparse
from tqdm import tqdm
from glob import glob
import smplx
import torch


def load_data_splits(csv_path):
    """Load data_splits.csv and return dict of recordings per split."""
    splits = {'train': [], 'val': [], 'test': []}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # ['train', 'val', 'test']
        for row in reader:
            for i, split_name in enumerate(header):
                if i < len(row) and row[i]:
                    splits[split_name].append(row[i])
    return splits


def load_pkl(pkl_path):
    """Load pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def create_smplx_model(model_path):
    """Load SMPL-X model for PCA hand components."""
    model = smplx.create(
        model_path=model_path,
        model_type='smplx',
        gender='neutral',
        use_pca=True,
        num_pca_comps=12,
        flat_hand_mean=False,
    )
    return model


def pca_to_full_hand_pose(pca_coeffs, hand_components, hand_mean):
    """
    Convert PCA hand pose (12-dim) to full hand pose (45-dim).

    Args:
        pca_coeffs: (12,) PCA coefficients
        hand_components: (12, 45) PCA components
        hand_mean: (45,) mean hand pose

    Returns:
        full_pose: (15, 3) full hand pose in axis-angle
    """
    pca_coeffs = torch.tensor(pca_coeffs, dtype=torch.float32).reshape(1, -1)
    full_pose = pca_coeffs @ hand_components + hand_mean  # (1, 45)
    return full_pose.reshape(15, 3).numpy()


def create_egobody_npz(egobody_dir, smplx_model_path, split, output_path, img_shape=(1080, 1920)):
    """
    Convert EgoBody PKL files to HumanData NPZ format.

    Args:
        egobody_dir: EgoBody dataset root path
        smplx_model_path: SMPL-X model path
        split: 'train', 'val', or 'test'
        output_path: output NPZ file path
        img_shape: (height, width)
    """

    # Load SMPL-X model for hand PCA components
    print("Loading SMPL-X model...")
    smplx_model = create_smplx_model(smplx_model_path)
    left_hand_components = smplx_model.left_hand_components  # (12, 45)
    right_hand_components = smplx_model.right_hand_components  # (12, 45)
    left_hand_mean = smplx_model.left_hand_mean  # (45,)
    right_hand_mean = smplx_model.right_hand_mean  # (45,)

    # Load split info
    splits_csv = osp.join(egobody_dir, 'data_splits.csv')
    splits = load_data_splits(splits_csv)
    recordings = splits[split]

    print(f"[{split}] {len(recordings)} recordings found")

    # Determine SMPLX folder based on split
    if split == 'train':
        smplx_dir = osp.join(egobody_dir, 'smplx_interactee_train')
    elif split == 'val':
        smplx_dir = osp.join(egobody_dir, 'smplx_interactee_val')
    else:  # test
        smplx_dir = osp.join(egobody_dir, 'smplx_interactee_test')

    if not osp.exists(smplx_dir):
        print(f"Error: {smplx_dir} does not exist!")
        return

    # Data collection
    all_data = {
        'image_path': [],
        'bbox_xywh': [],
        'lhand_bbox_xywh': [],
        'rhand_bbox_xywh': [],
        'face_bbox_xywh': [],
        # SMPL-X parameters
        'global_orient': [],
        'body_pose': [],
        'betas': [],
        'transl': [],
        'left_hand_pose': [],
        'right_hand_pose': [],
        'expression': [],
        'jaw_pose': [],
        # Keypoints
        'keypoints2d': [],
        'keypoints3d': [],
    }

    # Process each recording
    for recording in tqdm(recordings, desc=f"Processing {split}"):
        recording_path = osp.join(smplx_dir, recording)

        if not osp.exists(recording_path):
            continue

        # Find body_idx folders
        body_idx_folders = sorted(glob(osp.join(recording_path, 'body_idx_*')))

        for body_idx_folder in body_idx_folders:
            results_dir = osp.join(body_idx_folder, 'results')

            if not osp.exists(results_dir):
                continue

            # Process each frame
            frame_folders = sorted(glob(osp.join(results_dir, 'frame_*')))

            for frame_folder in frame_folders:
                frame_name = osp.basename(frame_folder)  # e.g., 'frame_01551'

                pkl_path = osp.join(frame_folder, '000.pkl')
                if not osp.exists(pkl_path):
                    continue

                try:
                    pkl_data = load_pkl(pkl_path)
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")
                    continue

                # Image path: kinect_color/{recording}/master/{frame_name}.jpg
                img_path = f"kinect_color/{recording}/master/{frame_name}.jpg"
                all_data['image_path'].append(img_path)

                # Bbox - full image [x, y, w, h, conf]
                h, w = img_shape
                all_data['bbox_xywh'].append(np.array([0, 0, w, h, 1.0], dtype=np.float32))

                # Hand/Face bbox - placeholder (conf=0)
                all_data['lhand_bbox_xywh'].append(np.array([0, 0, 0, 0, 0], dtype=np.float32))
                all_data['rhand_bbox_xywh'].append(np.array([0, 0, 0, 0, 0], dtype=np.float32))
                all_data['face_bbox_xywh'].append(np.array([0, 0, 0, 0, 0], dtype=np.float32))

                # SMPL-X parameters
                all_data['global_orient'].append(pkl_data['global_orient'].reshape(3).astype(np.float32))
                all_data['body_pose'].append(pkl_data['body_pose'].reshape(21, 3).astype(np.float32))
                all_data['betas'].append(pkl_data['betas'].reshape(10).astype(np.float32))
                all_data['transl'].append(pkl_data['transl'].reshape(3).astype(np.float32))
                all_data['expression'].append(pkl_data['expression'].reshape(10).astype(np.float32))
                all_data['jaw_pose'].append(pkl_data['jaw_pose'].reshape(3).astype(np.float32))

                # Hand pose: PCA (12,) -> Full (15, 3)
                left_pca = pkl_data['left_hand_pose'].reshape(12)
                right_pca = pkl_data['right_hand_pose'].reshape(12)

                left_hand_full = pca_to_full_hand_pose(left_pca, left_hand_components, left_hand_mean)
                right_hand_full = pca_to_full_hand_pose(right_pca, right_hand_components, right_hand_mean)

                all_data['left_hand_pose'].append(left_hand_full.astype(np.float32))
                all_data['right_hand_pose'].append(right_hand_full.astype(np.float32))

                # Keypoints - placeholder (zeros)
                all_data['keypoints2d'].append(np.zeros((144, 3), dtype=np.float32))
                all_data['keypoints3d'].append(np.zeros((144, 4), dtype=np.float32))

    # Convert to numpy arrays
    num_samples = len(all_data['image_path'])
    print(f"\nTotal samples: {num_samples}")

    if num_samples == 0:
        print("No samples found!")
        return

    # Create SMPLX dict
    smplx_dict = {
        'global_orient': np.stack(all_data['global_orient']),      # (N, 3)
        'body_pose': np.stack(all_data['body_pose']),              # (N, 21, 3)
        'betas': np.stack(all_data['betas']),                      # (N, 10)
        'transl': np.stack(all_data['transl']),                    # (N, 3)
        'left_hand_pose': np.stack(all_data['left_hand_pose']),    # (N, 15, 3)
        'right_hand_pose': np.stack(all_data['right_hand_pose']),  # (N, 15, 3)
        'expression': np.stack(all_data['expression']),            # (N, 10)
        'jaw_pose': np.stack(all_data['jaw_pose']),                # (N, 3)
    }

    # Keypoints mask (all zeros - placeholder)
    keypoints2d_mask = np.zeros(144, dtype=np.float32)
    keypoints3d_mask = np.zeros(144, dtype=np.float32)

    # Meta info
    meta = {
        'height': np.full(num_samples, img_shape[0]),
        'width': np.full(num_samples, img_shape[1]),
    }

    # Build output data
    output_data = {
        'image_path': np.array(all_data['image_path']),
        'bbox_xywh': np.stack(all_data['bbox_xywh']),
        'lhand_bbox_xywh': np.stack(all_data['lhand_bbox_xywh']),
        'rhand_bbox_xywh': np.stack(all_data['rhand_bbox_xywh']),
        'face_bbox_xywh': np.stack(all_data['face_bbox_xywh']),
        'smplx': smplx_dict,
        'keypoints2d': np.stack(all_data['keypoints2d']),
        'keypoints2d_mask': keypoints2d_mask,
        'keypoints3d': np.stack(all_data['keypoints3d']),
        'keypoints3d_mask': keypoints3d_mask,
        'meta': meta,
        '__keypoints_compressed__': False,
    }

    # Save NPZ
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    np.savez(output_path, **output_data)
    print(f"Saved to {output_path}")

    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"image_path: {output_data['image_path'].shape}")
    print(f"bbox_xywh: {output_data['bbox_xywh'].shape}")
    print(f"smplx.global_orient: {smplx_dict['global_orient'].shape}")
    print(f"smplx.body_pose: {smplx_dict['body_pose'].shape}")
    print(f"smplx.left_hand_pose: {smplx_dict['left_hand_pose'].shape}")
    print(f"smplx.right_hand_pose: {smplx_dict['right_hand_pose'].shape}")


def main():
    parser = argparse.ArgumentParser(description='Create EgoBody Kinect NPZ files')
    parser.add_argument('--egobody_dir', type=str, default='D:/Dev/Dataset/EgoBody',
                        help='EgoBody dataset root directory')
    parser.add_argument('--smplx_model_path', type=str,
                        default='common/utils/human_model_files',
                        help='SMPL-X model directory')
    parser.add_argument('--output_dir', type=str,
                        default='D:/Dev/Dataset/preprocessed_datasets',
                        help='Output directory for NPZ files')
    args = parser.parse_args()

    # Create all 3 splits
    split_mapping = {
        'train': 'egobody_kinect_train.npz',
        'val': 'egobody_kinect_valid.npz',  # val -> valid filename
        'test': 'egobody_kinect_test.npz',
    }

    for split, filename in split_mapping.items():
        print(f"\n{'='*50}")
        print(f"Creating {filename}...")
        print('='*50)

        output_path = osp.join(args.output_dir, filename)
        create_egobody_npz(
            args.egobody_dir,
            args.smplx_model_path,
            split,
            output_path
        )


if __name__ == '__main__':
    main()