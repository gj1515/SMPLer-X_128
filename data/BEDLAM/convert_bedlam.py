"""
Convert BEDLAM original NPZ files to SMPLer-X HumanData format.

Usage:
    python convert_bedlam.py \
        --input_dir D:/Dev/Dataset/BEDLAM/all_npz_12_training \
        --output_path D:/Dev/Dataset/preprocessed_datasets/bedlam_train.npz
"""

import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


# Constants
IMG_WIDTH = 1280
IMG_HEIGHT = 720
NUM_KEYPOINTS_SRC = 127  # BEDLAM original
NUM_KEYPOINTS_DST = 144  # HumanData format


def parse_args():
    parser = argparse.ArgumentParser(description='Convert BEDLAM to HumanData format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing BEDLAM NPZ files')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output HumanData NPZ path')
    return parser.parse_args()


def convert_single_npz(npz_path: str) -> dict:
    """Convert a single BEDLAM NPZ file to HumanData format.

    Args:
        npz_path: Path to BEDLAM NPZ file

    Returns:
        Converted data dictionary
    """
    data = np.load(npz_path, allow_pickle=True)
    npz_stem = os.path.splitext(os.path.basename(npz_path))[0]

    N = len(data['imgname'])

    # image_path: {npz_stem}/png/{imgname}
    image_path = np.array([f"{npz_stem}/png/{name}" for name in data['imgname']])

    # bbox_xywh from center/scale
    center = data['center']  # (N, 2)
    scale = data['scale']    # (N,)
    bbox_size = 200 * scale
    bbox_x = center[:, 0] - bbox_size / 2
    bbox_y = center[:, 1] - bbox_size / 2
    bbox_xywh = np.stack([
        bbox_x,
        bbox_y,
        bbox_size,
        bbox_size,
        np.ones(N)  # confidence
    ], axis=1).astype(np.float32)

    # keypoints2d: (N, 127, 3) -> (N, 144, 2) with zero padding
    gtkps = data['gtkps']  # (N, 127, 3) - [x, y, conf]
    keypoints2d = np.zeros((N, NUM_KEYPOINTS_DST, 2), dtype=np.float32)
    keypoints2d[:, :NUM_KEYPOINTS_SRC, :] = gtkps[:, :, :2]

    # pose_cam decomposition (165 params)
    pose_cam = data['pose_cam']  # (N, 165)
    global_orient = pose_cam[:, 0:3].astype(np.float32)                          # (N, 3)
    body_pose = pose_cam[:, 3:66].reshape(-1, 21, 3).astype(np.float32)          # (N, 21, 3)
    jaw_pose = pose_cam[:, 66:69].astype(np.float32)                             # (N, 3)
    # leye_pose = pose_cam[:, 69:72]  # not used
    # reye_pose = pose_cam[:, 72:75]  # not used
    left_hand_pose = pose_cam[:, 75:120].reshape(-1, 15, 3).astype(np.float32)   # (N, 15, 3)
    right_hand_pose = pose_cam[:, 120:165].reshape(-1, 15, 3).astype(np.float32) # (N, 15, 3)

    # shape (betas)
    betas = data['shape'][:, :10].astype(np.float32)  # (N, 10)

    # transl
    transl = data['trans_cam'].astype(np.float32)  # (N, 3)

    # expression (not available in BEDLAM, use zeros)
    expression = np.zeros((N, 10), dtype=np.float32)

    return {
        'image_path': image_path,
        'bbox_xywh': bbox_xywh,
        'keypoints2d': keypoints2d,
        'smplx': {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'jaw_pose': jaw_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'betas': betas,
            'transl': transl,
            'expression': expression,
        },
        'num_samples': N,
    }


def merge_data(data_list: list) -> dict:
    """Merge multiple converted data dictionaries into one.

    Args:
        data_list: List of converted data dictionaries

    Returns:
        Merged data dictionary
    """
    # Concatenate top-level arrays
    image_path = np.concatenate([d['image_path'] for d in data_list])
    bbox_xywh = np.concatenate([d['bbox_xywh'] for d in data_list])
    keypoints2d = np.concatenate([d['keypoints2d'] for d in data_list])

    # Concatenate smplx parameters
    smplx = {
        'global_orient': np.concatenate([d['smplx']['global_orient'] for d in data_list]),
        'body_pose': np.concatenate([d['smplx']['body_pose'] for d in data_list]),
        'jaw_pose': np.concatenate([d['smplx']['jaw_pose'] for d in data_list]),
        'left_hand_pose': np.concatenate([d['smplx']['left_hand_pose'] for d in data_list]),
        'right_hand_pose': np.concatenate([d['smplx']['right_hand_pose'] for d in data_list]),
        'betas': np.concatenate([d['smplx']['betas'] for d in data_list]),
        'transl': np.concatenate([d['smplx']['transl'] for d in data_list]),
        'expression': np.concatenate([d['smplx']['expression'] for d in data_list]),
    }

    total_samples = sum(d['num_samples'] for d in data_list)
    print(f'Total samples: {total_samples}')

    return {
        'image_path': image_path,
        'bbox_xywh': bbox_xywh,
        'keypoints2d': keypoints2d,
        'smplx': smplx,
        'num_samples': total_samples,
    }


def save_humandata(data: dict, output_path: str):
    """Save data in HumanData NPZ format.

    Args:
        data: Converted data dictionary
        output_path: Output file path
    """
    N = data['num_samples']

    # Create keypoints2d_mask: 127 valid, 17 zeros
    keypoints2d_mask = np.zeros(NUM_KEYPOINTS_DST, dtype=np.float32)
    keypoints2d_mask[:NUM_KEYPOINTS_SRC] = 1.0

    # Create meta dict
    meta = {
        'height': np.full(N, IMG_HEIGHT, dtype=np.int64),
        'width': np.full(N, IMG_WIDTH, dtype=np.int64),
    }

    # Build final data structure
    save_data = {
        'image_path': data['image_path'],
        'bbox_xywh': data['bbox_xywh'],
        'keypoints2d': data['keypoints2d'],
        'keypoints2d_mask': keypoints2d_mask,
        'smplx': data['smplx'],
        'meta': meta,
        '__keypoints_compressed__': False,
    }

    np.savez(output_path, **save_data)
    print(f'Saved {N} samples to {output_path}')


def main():
    args = parse_args()

    # Collect NPZ files
    npz_files = sorted(glob(os.path.join(args.input_dir, '*.npz')))
    print(f'Found {len(npz_files)} NPZ files')

    if len(npz_files) == 0:
        print('No NPZ files found.')
        return

    # Convert each file
    data_list = []
    for npz_path in tqdm(npz_files, desc='Converting'):
        data = convert_single_npz(npz_path)
        if data is not None:
            data_list.append(data)

    print(f'Converted {len(data_list)} files')

    # Merge data
    merged_data = merge_data(data_list)

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_humandata(merged_data, args.output_path)
    print(f'Saved to: {args.output_path}')


if __name__ == '__main__':
    main()