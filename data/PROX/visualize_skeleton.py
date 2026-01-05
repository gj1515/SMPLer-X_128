import json
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# Color camera intrinsics from PROX calibration/Color.json
FX, FY = 1060.531764702488, 1060.3856705041237
CX, CY = 951.2999547224418, 536.7703598373467

# Depth to Color translation (Color.T - IR.T)
# Kinect Depth and Color cameras are ~5cm apart
DEPTH_TO_COLOR_T = np.array([-0.052654542634615296, -1.861571860015632e-05, -8.75555283312419e-05])

# Kinect v2 skeleton bone connections
BONES = [
    # Spine
    ('SpineBase', 'SpineMid'),
    ('SpineMid', 'SpineShoulder'),
    ('SpineShoulder', 'Neck'),
    ('Neck', 'Head'),
    # Left arm
    ('SpineShoulder', 'ShoulderLeft'),
    ('ShoulderLeft', 'ElbowLeft'),
    ('ElbowLeft', 'WristLeft'),
    ('WristLeft', 'HandLeft'),
    ('HandLeft', 'HandTipLeft'),
    ('HandLeft', 'ThumbLeft'),
    # Right arm
    ('SpineShoulder', 'ShoulderRight'),
    ('ShoulderRight', 'ElbowRight'),
    ('ElbowRight', 'WristRight'),
    ('WristRight', 'HandRight'),
    ('HandRight', 'HandTipRight'),
    ('HandRight', 'ThumbRight'),
    # Left leg
    ('SpineBase', 'HipLeft'),
    ('HipLeft', 'KneeLeft'),
    ('KneeLeft', 'AnkleLeft'),
    ('AnkleLeft', 'FootLeft'),
    # Right leg
    ('SpineBase', 'HipRight'),
    ('HipRight', 'KneeRight'),
    ('KneeRight', 'AnkleRight'),
    ('AnkleRight', 'FootRight'),
]

# Joint colors by body part
JOINT_COLORS = {
    'Head': (0, 255, 255),      # Yellow
    'Neck': (0, 255, 255),
    'SpineShoulder': (0, 255, 0),  # Green
    'SpineMid': (0, 255, 0),
    'SpineBase': (0, 255, 0),
    'ShoulderLeft': (255, 0, 0),   # Blue (left)
    'ElbowLeft': (255, 0, 0),
    'WristLeft': (255, 0, 0),
    'HandLeft': (255, 128, 0),
    'HandTipLeft': (255, 128, 0),
    'ThumbLeft': (255, 128, 0),
    'ShoulderRight': (0, 0, 255),  # Red (right)
    'ElbowRight': (0, 0, 255),
    'WristRight': (0, 0, 255),
    'HandRight': (0, 128, 255),
    'HandTipRight': (0, 128, 255),
    'ThumbRight': (0, 128, 255),
    'HipLeft': (255, 0, 128),      # Purple (left leg)
    'KneeLeft': (255, 0, 128),
    'AnkleLeft': (255, 0, 128),
    'FootLeft': (255, 0, 128),
    'HipRight': (128, 0, 255),     # Magenta (right leg)
    'KneeRight': (128, 0, 255),
    'AnkleRight': (128, 0, 255),
    'FootRight': (128, 0, 255),
}


def project_3d_to_2d(X, Y, Z):
    """
    Project 3D point to 2D image coordinates.
    Direct perspective projection with Y flip.
    """
    u = FX * (X / Z) + CX
    v = FY * (-Y / Z) + CY  # Y flip (Kinect Y-up -> Image Y-down)

    return int(u), int(v)


def load_skeleton(json_path):
    """Load skeleton data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data['Bodies']:
        return None

    # Get first body
    joints = data['Bodies'][0]['Joints']

    # Project all joints to 2D
    joints_2d = {}
    for name, jdata in joints.items():
        if jdata['State'] == 'Tracked':
            X, Y, Z = jdata['Position']
            if Z > 0:  # Valid depth
                u, v = project_3d_to_2d(X, Y, Z)
                joints_2d[name] = (u, v)

    return joints_2d


def draw_skeleton(img, joints_2d):
    """Draw skeleton on image."""
    h, w = img.shape[:2]

    # Draw bones
    for j1, j2 in BONES:
        if j1 in joints_2d and j2 in joints_2d:
            pt1 = joints_2d[j1]
            pt2 = joints_2d[j2]
            # Check bounds
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                color = JOINT_COLORS.get(j1, (0, 255, 0))
                cv2.line(img, pt1, pt2, color, 3)

    # Draw joints
    for name, (u, v) in joints_2d.items():
        if 0 <= u < w and 0 <= v < h:
            color = JOINT_COLORS.get(name, (0, 255, 0))
            cv2.circle(img, (u, v), 6, color, -1)
            cv2.circle(img, (u, v), 6, (255, 255, 255), 2)

    return img


def process_sequence(color_dir, skeleton_dir, output_path, fps=30):
    """Process all frames and save as MP4 video."""
    # Get all color images
    color_files = sorted(glob(os.path.join(color_dir, '*.jpg')))

    if not color_files:
        print(f"No images found in {color_dir}")
        return

    # Get video dimensions from first image
    first_img = cv2.imread(color_files[0])
    h, w = first_img.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Processing {len(color_files)} frames...")

    for color_file in tqdm(color_files):
        # Load image
        img = cv2.imread(color_file)

        # Get corresponding skeleton file
        basename = os.path.splitext(os.path.basename(color_file))[0]
        skeleton_file = os.path.join(skeleton_dir, basename + '.json')

        if os.path.exists(skeleton_file):
            joints_2d = load_skeleton(skeleton_file)
            if joints_2d:
                img = draw_skeleton(img, joints_2d)

        out.write(img)

    out.release()
    print(f"Saved video to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize PROX skeleton on images')
    parser.add_argument('--recording', type=str, default=r'D:\Dev\Dataset\PROX\recordings\BasementSittingBooth_00142_01',)
    parser.add_argument('--output', type=str, default='skeleton.mp4', help='Output video path (default: recording_name_skeleton.mp4)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS (default: 30)')

    args = parser.parse_args()

    color_dir = os.path.join(args.recording, 'Color')
    skeleton_dir = os.path.join(args.recording, 'Skeleton')

    if args.output is None:
        recording_name = os.path.basename(args.recording)
        args.output = f"{recording_name}_skeleton.mp4"

    process_sequence(color_dir, skeleton_dir, args.output, args.fps)