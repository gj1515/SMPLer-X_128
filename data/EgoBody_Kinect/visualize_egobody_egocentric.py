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
                        left_hand_pose, right_hand_pose, jaw_pose, expression, gender
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Get gender from pkl (default to 'neutral' if not found)
    gender = data.get('gender', 'neutral')
    if isinstance(gender, np.ndarray):
        gender = str(gender)

    smplx_params = {
        'betas': data['betas'].reshape(-1),              # (10,)
        'global_orient': data['global_orient'].reshape(-1),  # (3,)
        'body_pose': data['body_pose'].reshape(-1),      # (63,)
        'transl': data['transl'].reshape(-1),            # (3,)
        'left_hand_pose': data.get('left_hand_pose', np.zeros((1, 12))).reshape(-1),
        'right_hand_pose': data.get('right_hand_pose', np.zeros((1, 12))).reshape(-1),
        'jaw_pose': data.get('jaw_pose', np.zeros((1, 3))).reshape(-1),
        'expression': data.get('expression', np.zeros((1, 10))).reshape(-1),
        'gender': gender,
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


def render_smplx(img, smplx_params, cam_param, smplx_layers, smplx_faces, render_func,
                 render_mesh=False, render_keypoints=True, debug=False):
    """Render SMPLX mesh and/or keypoints on image.

    Args:
        img: BGR image
        smplx_params: dict from load_smplx_pkl (includes 'gender')
        cam_param: dict with 'focal' and 'princpt'
        smplx_layers: dict of SMPLX model layers {'male', 'female', 'neutral'}
        smplx_faces: SMPLX faces
        render_func: render_mesh_on_image function
        render_mesh: whether to render mesh overlay
        render_keypoints: whether to render keypoints
        debug: print debug info

    Returns:
        img with SMPLX overlay
    """
    with torch.no_grad():
        # Get gender-specific layer
        gender = smplx_params.get('gender', 'neutral')
        if gender not in smplx_layers:
            gender = 'neutral'
        smplx_layer = smplx_layers[gender]

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

        # Render mesh (optional)
        if render_mesh:
            vertices = output.vertices[0].cpu().numpy()
            img = render_func(img, vertices, smplx_faces, cam_param)

        # Render keypoints (optional)
        if render_keypoints:
            joints = output.joints[0].cpu().numpy()  # (num_joints, 3)
            focal = cam_param['focal']
            princpt = cam_param['princpt']

            if debug:
                print(f"joints shape: {joints.shape}")
                print(f"joints[0] (pelvis): {joints[0]}")
                print(f"Z range: {joints[:, 2].min():.3f} ~ {joints[:, 2].max():.3f}")

            # Project 3D joints to 2D
            for joint in joints:
                if joint[2] > 0:  # Only project if Z > 0 (in front of camera)
                    x = int(focal[0] * joint[0] / joint[2] + princpt[0])
                    y = int(focal[1] * joint[1] / joint[2] + princpt[1])
                    if debug:
                        print(f"  projected: ({x}, {y})")
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    return img


def main(args):
    # Scan images from seq_path (kinect_color/.../master/)
    img_extensions = ('.jpg', '.jpeg', '.png')
    img_files = sorted([f for f in os.listdir(args.seq_path)
                        if f.lower().endswith(img_extensions)])

    if not img_files:
        raise FileNotFoundError(f"No images found in {args.seq_path}")

    # Fixed camera parameters (Kinect)
    cam_param = {
        'focal': [918.241638, 918.177368],
        'princpt': [958.487976, 551.059509]
    }

    # Load SMPLX model if smplx_path provided
    smplx_layers = None
    smplx_faces = None
    render_func = None

    if args.smplx_path:
        print("Loading SMPLX model...")
        from utils.human_models import smpl_x
        from utils.vis import render_mesh_on_image

        smplx_layers = smpl_x.layer  # dict with 'male', 'female', 'neutral'
        smplx_faces = smpl_x.face
        render_func = render_mesh_on_image

    # Get image size from first image
    first_img_path = osp.join(args.seq_path, img_files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        raise FileNotFoundError(f"Cannot read image: {first_img_path}")
    h, w = first_img.shape[:2]

    # Setup video writer
    os.makedirs(osp.dirname(args.output) if osp.dirname(args.output) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    print(f"Processing {len(img_files)} frames...")
    print(f"Image size: {w}x{h}")
    print(f"Output: {args.output}")

    for img_file in tqdm(img_files):
        img_path = osp.join(args.seq_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Cannot read {img_path}")
            continue

        # Extract frame name (e.g., frame_00001.jpg -> frame_00001)
        frame = osp.splitext(img_file)[0]

        # Render SMPLX joints if available
        if args.smplx_path:
            pkl_path = osp.join(args.smplx_path, frame, '000.pkl')
            if osp.exists(pkl_path):
                smplx_params = load_smplx_pkl(pkl_path)
                # Debug: print first frame info
                is_first = (img_file == img_files[0])
                if is_first:
                    print(f"Found pkl: {pkl_path}")
                    print(f"gender: {smplx_params['gender']}")
                    print(f"transl: {smplx_params['transl']}")
                    print(f"global_orient: {smplx_params['global_orient']}")
                # No coordinate transform needed - SMPLX is already in Kinect coordinates
                img = render_smplx(img, smplx_params, cam_param, smplx_layers, smplx_faces, render_func,
                                   render_mesh=args.render_mesh, render_keypoints=args.render_keypoints,
                                   debug=is_first)
            elif img_file == img_files[0]:
                print(f"pkl not found: {pkl_path}")

        out.write(img)

    out.release()
    print(f"Done! Saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=r'D:\Dev\Dataset\EgoBody',
                        help='Root path to EgoBody dataset')
    parser.add_argument('--seq_path', type=str,
                        default=r'D:\Dev\Dataset\EgoBody\kinect_color\recording_20210911_S03_S08_01\master',
                        help='Path to sequence folder containing keypoints.npz and valid_frame.npz')
    parser.add_argument('--smplx_path', type=str,
                        default=r'D:\Dev\Dataset\EgoBody\smplx_interactee_test\recording_20210911_S03_S08_01\body_idx_1\results',
                        help='Path to SMPLX results folder (e.g., smplx_camera_wearer_val/.../results)')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    parser.add_argument('--render_mesh', action='store_true', default=True,
                        help='Render SMPLX mesh overlay')
    parser.add_argument('--render_keypoints', action='store_true', dest='render_keypoints', default=True,
                        help='Render SMPLX keypoints (default: True)')
    parser.add_argument('--no-render_keypoints', action='store_false', dest='render_keypoints',
                        help='Disable SMPLX keypoints rendering')
    args = parser.parse_args()

    main(args)