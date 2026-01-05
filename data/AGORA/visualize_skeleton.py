"""
AGORA Skeleton Visualization Script
Visualize 2D keypoints and bones on AGORA images.

Usage:
    python visualize_skeleton.py \
        --img_path /path/to/image.png \
        --joints_path /path/to/joints_2d.json \
        --output_dir skeleton_vis

Added by SH Heo (260105)
"""

import argparse
import os
import cv2
import numpy as np
import json


def visualize_skeleton(img_path, joints_path, output_dir='skeleton_vis'):
    """
    Visualize 2D keypoints and bones on a single AGORA image.

    Args:
        img_path: Path to input image
        joints_path: Path to 2D joints JSON file
        output_dir: Output directory for visualized image
    """
    os.makedirs(output_dir, exist_ok=True)

    # AGORA joint parts (127 joints total)
    joint_parts = {
        'body': (list(range(0, 25)) + list(range(55, 66)), (0, 0, 255)),      # Red
        'lhand': (list(range(25, 40)) + list(range(66, 71)), (0, 255, 0)),    # Green
        'rhand': (list(range(40, 55)) + list(range(71, 76)), (255, 0, 0)),    # Blue
        'face': (list(range(76, 127)), (0, 255, 255))                          # Yellow
    }

    # Bone connections for body, left hand, right hand (face: dots only)
    body_bones = [
        (0, 1), (0, 2), (0, 3),  # Pelvis -> L_Hip, R_Hip, Spine_1
        (1, 4), (4, 7), (7, 10), (7, 62),  # Left leg
        (2, 5), (5, 8), (8, 11), (8, 65),  # Right leg
        (10, 60), (10, 61), (11, 63), (11, 64),  # Toes
        (3, 6), (6, 9), (9, 12),  # Spine
        (12, 13), (12, 14), (12, 15),  # Neck -> Collars, Head
        (13, 16), (16, 18), (18, 20),  # Left arm
        (14, 17), (17, 19), (19, 21),  # Right arm
        (15, 55), (55, 56), (55, 57), (56, 58), (57, 59),  # Head -> Nose -> Eyes -> Ears
    ]

    # Left hand bones: L_Wrist(20) -> fingers
    lhand_bones = [
        (20, 25), (25, 26), (26, 27), (27, 67),  # Index
        (20, 28), (28, 29), (29, 30), (30, 68),  # Middle
        (20, 31), (31, 32), (32, 33), (33, 70),  # Pinky
        (20, 34), (34, 35), (35, 36), (36, 69),  # Ring
        (20, 37), (37, 38), (38, 39), (39, 66),  # Thumb
    ]

    # Right hand bones: R_Wrist(21) -> fingers
    rhand_bones = [
        (21, 40), (40, 41), (41, 42), (42, 72),  # Index
        (21, 43), (43, 44), (44, 45), (45, 73),  # Middle
        (21, 46), (46, 47), (47, 48), (48, 75),  # Pinky
        (21, 49), (49, 50), (50, 51), (51, 74),  # Ring
        (21, 52), (52, 53), (53, 54), (54, 71),  # Thumb
    ]

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f'[ERROR] Failed to load image: {img_path}')
        return

    # Load 2D joints from JSON (original coords are in 3840x2160)
    if not os.path.isfile(joints_path):
        print(f'[ERROR] Joints file not found: {joints_path}')
        return

    with open(joints_path) as f:
        joint_img = np.array(json.load(f)).reshape(-1, 2)

    # Get image dimensions
    img_h, img_w = img.shape[:2]

    # Scale joints from 3840x2160 to current image resolution
    joint_img[:, 0] = joint_img[:, 0] / 3840 * img_w
    joint_img[:, 1] = joint_img[:, 1] / 2160 * img_h

    # Helper function to draw bones
    def draw_bones(bones, color):
        for (j1, j2) in bones:
            if j1 < len(joint_img) and j2 < len(joint_img):
                x1, y1 = joint_img[j1]
                x2, y2 = joint_img[j2]
                if (0 <= x1 < img_w and 0 <= y1 < img_h and
                    0 <= x2 < img_w and 0 <= y2 < img_h):
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    # Draw bones first (so joints are drawn on top)
    draw_bones(body_bones, joint_parts['body'][1])
    draw_bones(lhand_bones, joint_parts['lhand'][1])
    draw_bones(rhand_bones, joint_parts['rhand'][1])

    # Draw keypoints by body part
    for part_name, (indices, color) in joint_parts.items():
        for j_idx in indices:
            if j_idx < len(joint_img):
                x_kp, y_kp = joint_img[j_idx]
                if 0 <= x_kp < img_w and 0 <= y_kp < img_h:
                    cv2.circle(img, (int(x_kp), int(y_kp)), 2, color, -1)

    # Add legend
    legend_y = 30
    for part_name, (_, color) in joint_parts.items():
        cv2.putText(img, part_name, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        legend_y += 25

    # Save image
    img_name = os.path.basename(img_path)
    save_path = os.path.join(output_dir, f'skeleton_{img_name}')
    cv2.imwrite(save_path, img)
    print(f'[INFO] Saved visualization to: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize AGORA skeleton on image')
    parser.add_argument('--img_path', required=True, help='Path to input image')
    parser.add_argument('--joints_path', required=True, help='Path to 2D joints JSON file')
    parser.add_argument('--output_dir', default='skeleton_vis', help='Output directory')
    args = parser.parse_args()

    visualize_skeleton(args.img_path, args.joints_path, args.output_dir)


if __name__ == '__main__':
    main()
