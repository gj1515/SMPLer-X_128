import os
import sys
import os.path as osp
import numpy as np
import cv2
import pickle
import torch
import json
import re
from tqdm import tqdm
import argparse
from scipy.spatial.transform import Rotation

# Add project paths for imports
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..', 'main'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..', 'common'))


# OpenPose 25 keypoints bone connections
OPENPOSE_25_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # right arm
    (1, 5), (5, 6), (6, 7),              # left arm
    (1, 8),                               # spine
    (8, 9), (9, 10), (10, 11),           # right leg
    (8, 12), (12, 13), (13, 14),         # left leg
    (0, 15), (15, 17),                   # right eye/ear
    (0, 16), (16, 18),                   # left eye/ear
    (11, 24), (11, 22), (22, 23),        # right foot
    (14, 21), (14, 19), (19, 20),        # left foot
]


def parse_pv_txt(pv_txt_path):
    """Parse pv.txt to get camera parameters and pv2world transforms.

    Returns:
        cx, cy: principal point (fixed)
        cam_params: dict {timestamp: {'focal': (fx, fy), 'pv2world': 4x4 matrix}}
    """
    cam_params = {}
    with open(pv_txt_path, 'r') as f:
        lines = f.readlines()

    # First line: cx, cy, width, height
    first_line = lines[0].strip().split(',')
    cx, cy = float(first_line[0]), float(first_line[1])

    # Remaining lines: timestamp, fx, fy, pv2world (16 values)
    for line in lines[1:]:
        parts = line.strip().split(',')
        timestamp = parts[0]
        fx, fy = float(parts[1]), float(parts[2])

        # Parse pv2world transform (4x4 matrix)
        if len(parts) >= 19:  # timestamp + fx + fy + 16 matrix values
            pv2world = np.array([float(x) for x in parts[3:19]]).reshape(4, 4)
        else:
            pv2world = np.eye(4)

        cam_params[timestamp] = {
            'focal': (fx, fy),
            'pv2world': pv2world
        }

    return cx, cy, cam_params


def load_holo_to_kinect(calibration_path):
    """Load holo_to_kinect transformation from calibration json.

    Returns:
        4x4 transformation matrix (HoloLens world -> Kinect)
    """
    with open(calibration_path, 'r') as f:
        data = json.load(f)
    return np.array(data['trans'])


def transform_smplx_params(smplx_params, transform_matrix):
    """Transform SMPLX global_orient and transl to new coordinate system.

    Args:
        smplx_params: dict with 'global_orient' (axis-angle) and 'transl'
        transform_matrix: 4x4 transformation matrix

    Returns:
        transformed smplx_params (copy)
    """
    # Get rotation and translation from transform matrix
    T_rot = transform_matrix[:3, :3]
    T_trans = transform_matrix[:3, 3]

    # Transform global_orient (axis-angle -> rotation matrix -> transform -> axis-angle)
    global_orient_aa = smplx_params['global_orient']  # (3,)
    R_orig = Rotation.from_rotvec(global_orient_aa).as_matrix()
    R_new = T_rot @ R_orig
    global_orient_new = Rotation.from_matrix(R_new).as_rotvec()

    # Transform transl
    transl_orig = smplx_params['transl']  # (3,)
    transl_new = T_rot @ transl_orig + T_trans

    # Update params
    smplx_params_new = smplx_params.copy()
    smplx_params_new['global_orient'] = global_orient_new.astype(np.float32)
    smplx_params_new['transl'] = transl_new.astype(np.float32)

    return smplx_params_new


def load_smplx_pkl(pkl_path):
    """Load SMPLX parameters from pkl file.

    Returns:
        dict with keys: betas, global_orient, body_pose, transl,
                        left_hand_pose, right_hand_pose, jaw_pose, expression
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    smplx_params = {
        'betas': data['betas'].reshape(-1),              # (10,)
        'global_orient': data['global_orient'].reshape(-1),  # (3,)
        'body_pose': data['body_pose'].reshape(-1),      # (63,)
        'transl': data['transl'].reshape(-1),            # (3,)
        'left_hand_pose': data.get('left_hand_pose', np.zeros((1, 12))).reshape(-1),
        'right_hand_pose': data.get('right_hand_pose', np.zeros((1, 12))).reshape(-1),
        'jaw_pose': data.get('jaw_pose', np.zeros((1, 3))).reshape(-1),
        'expression': data.get('expression', np.zeros((1, 10))).reshape(-1),
    }
    return smplx_params


def extract_timestamp_frame(imgname):
    """Extract timestamp and frame from imgname.

    Example:
        imgname: egocentric_color/.../PV/132767028221338608_frame_01111.jpg
        returns: ('132767028221338608', 'frame_01111')
    """
    basename = osp.basename(imgname)  # 132767028221338608_frame_01111.jpg
    name = osp.splitext(basename)[0]  # 132767028221338608_frame_01111
    parts = name.split('_frame_')
    timestamp = parts[0]
    frame = 'frame_' + parts[1]
    return timestamp, frame


def center_scale_to_bbox(center, scale, scale_factor=200):
    """Convert center and scale to bbox [x1, y1, x2, y2]"""
    bbox_size = scale * scale_factor
    x1 = center[0] - bbox_size / 2
    y1 = center[1] - bbox_size / 2
    x2 = center[0] + bbox_size / 2
    y2 = center[1] + bbox_size / 2
    return [int(x1), int(y1), int(x2), int(y2)]


def draw_keypoints(img, keypoints, color=(0, 0, 255), radius=4, conf_thresh=0.3):
    """Draw 2D keypoints on image"""
    for kpt in keypoints:
        x, y, conf = kpt
        if conf > conf_thresh:
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
    return img


def draw_bones(img, keypoints, bones, color=(0, 0, 255), thickness=2, conf_thresh=0.3):
    """Draw bone connections on image"""
    for (i, j) in bones:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 > conf_thresh and c2 > conf_thresh:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img


def render_smplx(img, smplx_params, cam_param, smplx_layer, smplx_faces, render_func):
    """Render SMPLX mesh on image.

    Args:
        img: BGR image
        smplx_params: dict from load_smplx_pkl
        cam_param: dict with 'focal' and 'princpt'
        smplx_layer: SMPLX model layer
        smplx_faces: SMPLX faces
        render_func: render_mesh_on_image function

    Returns:
        img with SMPLX mesh overlay
    """
    with torch.no_grad():
        # Prepare parameters
        betas = torch.FloatTensor(smplx_params['betas']).unsqueeze(0)
        global_orient = torch.FloatTensor(smplx_params['global_orient']).unsqueeze(0)
        body_pose = torch.FloatTensor(smplx_params['body_pose']).unsqueeze(0)
        transl = torch.FloatTensor(smplx_params['transl']).unsqueeze(0)

        # Hand pose: pkl has 12 dims (PCA), need to convert or use zeros for 45 dims
        left_hand_pose = torch.zeros(1, 45)
        right_hand_pose = torch.zeros(1, 45)

        jaw_pose = torch.FloatTensor(smplx_params['jaw_pose']).unsqueeze(0)
        expression = torch.FloatTensor(smplx_params['expression']).unsqueeze(0)

        # Forward SMPLX
        output = smplx_layer(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=torch.zeros(1, 3),
            reye_pose=torch.zeros(1, 3),
            expression=expression,
            transl=transl
        )

        vertices = output.vertices[0].cpu().numpy()

        # Render mesh on image
        img = render_func(img, vertices, smplx_faces, cam_param)

    return img


def main(args):
    # Load data
    keypoints_path = osp.join(args.seq_path, 'keypoints.npz')
    valid_frame_path = osp.join(args.seq_path, 'valid_frame.npz')

    kp_data = np.load(keypoints_path, allow_pickle=True)
    vf_data = np.load(valid_frame_path, allow_pickle=True)

    # Get valid frames
    valid_mask = vf_data['valid']
    valid_imgnames = vf_data['imgname'][valid_mask]

    # Build keypoints lookup
    kp_imgnames = kp_data['imgname']
    kp_centers = kp_data['center']
    kp_scales = kp_data['scale']
    kp_keypoints = kp_data['keypoints']  # (N, 25, 3)

    kp_lookup = {name: i for i, name in enumerate(kp_imgnames)}

    # Load SMPLX model and camera params if smplx_path provided
    smplx_layer = None
    smplx_faces = None
    render_func = None
    cx, cy = None, None
    pv_cam_params = None

    # Extract recording name from seq_path for calibration
    # seq_path: .../egocentric_color/recording_XXXXXX/2021-XX-XX-XXXXXX
    holo_to_kinect = None

    if args.smplx_path:
        print("Loading SMPLX model...")
        from utils.human_models import smpl_x
        from utils.vis import render_mesh_on_image

        smplx_layer = smpl_x.layer['neutral']
        smplx_faces = smpl_x.face
        render_func = render_mesh_on_image

        # Load camera parameters from pv.txt
        seq_folder_name = osp.basename(args.seq_path)
        pv_txt_path = osp.join(args.seq_path, f'{seq_folder_name}_pv.txt')
        if osp.exists(pv_txt_path):
            cx, cy, pv_cam_params = parse_pv_txt(pv_txt_path)
            print(f"Loaded camera params from {pv_txt_path}")
            print(f"Principal point: cx={cx}, cy={cy}")
            print(f"Number of timestamps: {len(pv_cam_params)}")
        else:
            raise FileNotFoundError(f"pv.txt not found at {pv_txt_path}")

        # Extract recording name and load holo_to_kinect calibration
        # Path: egocentric_color/recording_XXXXX/2021-XX-XX -> recording_XXXXX
        recording_name = osp.basename(osp.dirname(args.seq_path))
        calibration_path = osp.join(args.data_root, 'calibrations', recording_name, 'cal_trans', 'holo_to_kinect12.json')
        if osp.exists(calibration_path):
            holo_to_kinect = load_holo_to_kinect(calibration_path)
            print(f"Loaded holo_to_kinect from {calibration_path}")
        else:
            print(f"Warning: Calibration not found at {calibration_path}, skipping coordinate transform")

    # Get image size from first image
    first_img_path = osp.join(args.data_root, valid_imgnames[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        raise FileNotFoundError(f"Cannot read image: {first_img_path}")
    h, w = first_img.shape[:2]

    # Setup video writer
    os.makedirs(osp.dirname(args.output) if osp.dirname(args.output) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    print(f"Processing {len(valid_imgnames)} valid frames...")
    print(f"Image size: {w}x{h}")
    print(f"Output: {args.output}")

    for imgname in tqdm(valid_imgnames):
        # Get image path
        img_path = osp.join(args.data_root, imgname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Cannot read {img_path}")
            continue

        # Extract timestamp and frame from imgname
        timestamp, frame = extract_timestamp_frame(imgname)

        # Render SMPLX mesh if available
        if args.smplx_path and cx is not None:
            pkl_path = osp.join(args.smplx_path, frame, '000.pkl')
            if osp.exists(pkl_path):
                smplx_params = load_smplx_pkl(pkl_path)

                # Get per-frame camera parameters from pv.txt
                if pv_cam_params is not None and timestamp in pv_cam_params:
                    frame_data = pv_cam_params[timestamp]
                    fx, fy = frame_data['focal']
                    pv2world = frame_data['pv2world']
                else:
                    # Use first available if timestamp not found
                    first_data = list(pv_cam_params.values())[0]
                    fx, fy = first_data['focal']
                    pv2world = first_data['pv2world']

                # Transform SMPLX from Kinect to HoloLens camera coordinates
                # Transform chain: Kinect -> HoloLens world -> HoloLens camera
                if holo_to_kinect is not None:
                    kinect_to_holo_world = np.linalg.inv(holo_to_kinect)
                    holo_world_to_holo_cam = np.linalg.inv(pv2world)
                    transform = holo_world_to_holo_cam @ kinect_to_holo_world
                    smplx_params = transform_smplx_params(smplx_params, transform)

                cam_param = {
                    'focal': [fx, fy],
                    'princpt': [cx, cy]
                }
                img = render_smplx(img, smplx_params, cam_param, smplx_layer, smplx_faces, render_func)

        # Get bbox and keypoints from data
        if imgname in kp_lookup:
            idx = kp_lookup[imgname]
            center = kp_centers[idx]
            scale = kp_scales[idx]
            keypoints = kp_keypoints[idx]  # (25, 3)

            # Convert to bbox
            x1, y1, x2, y2 = center_scale_to_bbox(center, scale)

            # Draw green bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw red bones and keypoints
            img = draw_bones(img, keypoints, OPENPOSE_25_BONES, color=(0, 0, 255), thickness=2)
            img = draw_keypoints(img, keypoints, color=(0, 0, 255), radius=4)

        out.write(img)

    out.release()
    print(f"Done! Saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=r'D:\Dev\Dataset\EgoBody',
                        help='Root path to EgoBody dataset')
    parser.add_argument('--seq_path', type=str,
                        default=r'D:\Dev\Dataset\EgoBody\egocentric_color\recording_20210921_S11_S10_01\2021-09-21-145953',
                        help='Path to sequence folder containing keypoints.npz and valid_frame.npz')
    parser.add_argument('--smplx_path', type=str,
                        default=r'D:\Dev\Dataset\EgoBody\smplx_interactee_val\recording_20210921_S11_S10_01\body_idx_1\results',
                        help='Path to SMPLX results folder (e.g., smplx_camera_wearer_val/.../results)')
    parser.add_argument('--output', type=str, default='output_bbox.mp4',
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    args = parser.parse_args()

    main(args)