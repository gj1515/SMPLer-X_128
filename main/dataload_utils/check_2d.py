import cv2
import numpy as np
from config import cfg

# Added by SH Heo(260105)
def show_input_image(inputs, window_name='Input Image', wait_key=1):
    img_tensor = inputs['img']

    # [C, H, W]
    img = img_tensor[0].cpu().numpy()

    # [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)

    # Restore normalization (if 0~1 range, convert to 0~255)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # RGB -> BGR for O
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow(window_name, img)

    key = cv2.waitKey(wait_key)
    if key == 27:
        cv2.destroyWindow(window_name)

def draw_2d_wholebody_kpts(inputs, targets, meta_info, window_name='2D Keypoints', wait_key=0):
    """
    Draw 2D wholebody keypoints on input image.

    Args:
        inputs: {'img': [B,C,H,W] tensor}
        targets: {'joint_img': [B,J,3] tensor}
        meta_info: {'joint_valid': [B,J,1] tensor}
        window_name: window title
        wait_key: 0 = wait for key, ESC to close
    """
    # 1. Image: tensor -> numpy
    img_tensor = inputs['img']
    img = img_tensor[0].cpu().numpy()  # [C, H, W]
    img = img.transpose(1, 2, 0)  # [H, W, C]

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.copy()  # make writable

    # 2. Get joint coordinates and validity
    joint_img = targets['joint_img'][0].cpu().numpy()  # [137, 3]
    joint_valid = meta_info['joint_valid'][0].cpu().numpy()  # [137, 1]

    # 3. Convert coordinates: output_hm_shape -> input_img_shape
    # joint_img is in output_hm_shape space (16, 16, 12)
    # need to scale to input_img_shape (512, 384)
    joint_img_scaled = joint_img.copy()
    joint_img_scaled[:, 0] = joint_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1] # joint_img[0] / 12 * 384
    joint_img_scaled[:, 1] = joint_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0] # joint_img[1] / 16 * 512

    # 4. Define body part ranges and colors (BGR) - SMPLX 137 joints order
    body_parts = {
        'body': (range(0, 25), (0, 0, 255)),      # Red (0-24)
        'lhand': (range(25, 45), (0, 255, 0)),    # Green (25-44)
        'rhand': (range(45, 65), (255, 0, 0)),    # Blue (45-64)
        'face': (range(65, 137), (0, 255, 255)),  # Yellow (65-136)
    }

    # 5. Define skeleton connections (SMPLX 137 joints order)
    # Body: 0-Pelvis, 1-L_Hip, 2-R_Hip, 3-L_Knee, 4-R_Knee, 5-L_Ankle, 6-R_Ankle,
    #       7-Neck, 8-L_Shoulder, 9-R_Shoulder, 10-L_Elbow, 11-R_Elbow, 12-L_Wrist, 13-R_Wrist,
    #       14-L_Big_toe, 15-L_Small_toe, 16-L_Heel, 17-R_Big_toe, 18-R_Small_toe, 19-R_Heel,
    #       20-L_Ear, 21-R_Ear, 22-L_Eye, 23-R_Eye, 24-Nose
    body_skeleton = [
        (0, 1), (0, 2),      # Pelvis -> L_Hip, R_Hip
        (1, 3), (3, 5),      # L_Hip -> L_Knee -> L_Ankle
        (2, 4), (4, 6),      # R_Hip -> R_Knee -> R_Ankle
        (0, 7),              # Pelvis -> Neck
        (7, 8), (7, 9),      # Neck -> L_Shoulder, R_Shoulder
        (8, 10), (10, 12),   # L_Shoulder -> L_Elbow -> L_Wrist
        (9, 11), (11, 13),   # R_Shoulder -> R_Elbow -> R_Wrist
        (5, 14), (5, 15), (5, 16),  # L_Ankle -> L_Big_toe, L_Small_toe, L_Heel
        (6, 17), (6, 18), (6, 19),  # R_Ankle -> R_Big_toe, R_Small_toe, R_Heel
        (7, 24),             # Neck -> Nose
        (24, 22), (24, 23),  # Nose -> L_Eye, R_Eye
        (22, 20), (23, 21),  # L_Eye -> L_Ear, R_Eye -> R_Ear
    ]

    # Left Hand: from L_Wrist(12), 25-28 L_Thumb, 29-32 L_Index, 33-36 L_Middle, 37-40 L_Ring, 41-44 L_Pinky
    lhand_skeleton = [
        (12, 25), (25, 26), (26, 27), (27, 28),  # L_Wrist -> Thumb
        (12, 29), (29, 30), (30, 31), (31, 32),  # L_Wrist -> Index
        (12, 33), (33, 34), (34, 35), (35, 36),  # L_Wrist -> Middle
        (12, 37), (37, 38), (38, 39), (39, 40),  # L_Wrist -> Ring
        (12, 41), (41, 42), (42, 43), (43, 44),  # L_Wrist -> Pinky
    ]

    # Right Hand: from R_Wrist(13), 45-48 R_Thumb, 49-52 R_Index, 53-56 R_Middle, 57-60 R_Ring, 61-64 R_Pinky
    rhand_skeleton = [
        (13, 45), (45, 46), (46, 47), (47, 48),  # R_Wrist -> Thumb
        (13, 49), (49, 50), (50, 51), (51, 52),  # R_Wrist -> Index
        (13, 53), (53, 54), (54, 55), (55, 56),  # R_Wrist -> Middle
        (13, 57), (57, 58), (58, 59), (59, 60),  # R_Wrist -> Ring
        (13, 61), (61, 62), (62, 63), (63, 64),  # R_Wrist -> Pinky
    ]

    # Face contour: 120-136 (Face_56 ~ Face_72, 17 points)
    face_contour_skeleton = [(i, i+1) for i in range(120, 136)]

    # 6. Draw bones first (so keypoints are on top)
    skeleton_parts = {
        'body': (body_skeleton, (0, 0, 255)),      # Red
        'lhand': (lhand_skeleton, (0, 255, 0)),    # Green
        'rhand': (rhand_skeleton, (255, 0, 0)),    # Blue
        'face': (face_contour_skeleton, (0, 255, 255)),  # Yellow
    }

    for part_name, (skeleton, color) in skeleton_parts.items():
        for (i, j) in skeleton:
            if i >= len(joint_valid) or j >= len(joint_valid):
                continue
            if joint_valid[i, 0] > 0 and joint_valid[j, 0] > 0:
                x1, y1 = int(joint_img_scaled[i, 0]), int(joint_img_scaled[i, 1])
                x2, y2 = int(joint_img_scaled[j, 0]), int(joint_img_scaled[j, 1])
                # Check bounds
                if (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0] and
                    0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]):
                    cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 7. Draw keypoints
    for part_name, (indices, color) in body_parts.items():
        for idx in indices:
            if idx >= len(joint_valid):
                continue
            if joint_valid[idx, 0] > 0:
                x = int(joint_img_scaled[idx, 0])
                y = int(joint_img_scaled[idx, 1])
                # Check if within image bounds
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    radius = 2
                    cv2.circle(img, (x, y), radius, color, -1)

    # 8. Draw hand/face bounding boxes
    bbox_parts = {
        'lhand': ((0, 255, 0), 'lhand_bbox_center', 'lhand_bbox_size', 'lhand_bbox_valid'),
        'rhand': ((255, 0, 0), 'rhand_bbox_center', 'rhand_bbox_size', 'rhand_bbox_valid'),
        'face': ((0, 255, 255), 'face_bbox_center', 'face_bbox_size', 'face_bbox_valid'),
    }

    for part_name, (color, center_key, size_key, valid_key) in bbox_parts.items():
        # Check if keys exist in targets/meta_info
        if center_key not in targets or size_key not in targets or valid_key not in meta_info:
            continue

        bbox_valid = meta_info[valid_key][0].item()
        if bbox_valid <= 0:
            continue

        # Get center and size (in output_hm_shape space)
        center = targets[center_key][0].cpu().numpy()  # [2]
        size = targets[size_key][0].cpu().numpy()      # [2]

        # Scale to input_img_shape
        cx = center[0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        cy = center[1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        w = size[0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        h = size[1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]

        # Calculate corners
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)

        # Clamp to image bounds
        x1 = max(0, min(x1, img.shape[1] - 1))
        y1 = max(0, min(y1, img.shape[0] - 1))
        x2 = max(0, min(x2, img.shape[1] - 1))
        y2 = max(0, min(y2, img.shape[0] - 1))

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    cv2.imshow(window_name, img)
    key = cv2.waitKey(wait_key)

    if key == 27:
        cv2.destroyWindow(window_name)


def draw_projected_3d_joints(inputs, model_output, window_name='Projected 3D Joints', wait_key=0):
    """
    Draw projected 3D joints (from SMPL-X model output) on input image.

    Args:
        inputs: {'img': [B,C,H,W] tensor}
        model_output: {'joint_proj': [B,137,2] tensor} - projected joints in output_hm_shape space
        window_name: window title
        wait_key: 0 = wait for key, ESC to close
    """
    # 1. Image: tensor -> numpy
    img_tensor = inputs['img']
    img = img_tensor[0].cpu().numpy()  # [C, H, W]
    img = img.transpose(1, 2, 0)  # [H, W, C]

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.copy()  # make writable

    # 2. Get projected joint coordinates
    joint_proj = model_output['joint_proj'][0].cpu().numpy()  # [137, 2]

    # 3. Convert coordinates: output_hm_shape -> input_img_shape
    # joint_proj is in output_hm_shape space (16, 16, 12)
    # need to scale to input_img_shape (512, 384)
    joint_proj_scaled = joint_proj.copy()
    joint_proj_scaled[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]  # x: /12 * 384
    joint_proj_scaled[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]  # y: /16 * 512

    # 4. Define body part ranges and colors (BGR) - SMPLX 137 joints order
    body_parts = {
        'body': (range(0, 25), (0, 255, 255)),      # Yellow - body (0-24)
        'lhand': (range(25, 45), (0, 255, 0)),      # Green - left hand (25-44)
        'rhand': (range(45, 65), (255, 0, 0)),      # Blue - right hand (45-64)
        'face': (range(65, 137), (255, 0, 255)),    # Magenta - face (65-136)
    }

    # 5. Define skeleton connections (same as draw_2d_wholebody_kpts)
    body_skeleton = [
        (0, 1), (0, 2),      # Pelvis -> L_Hip, R_Hip
        (1, 3), (3, 5),      # L_Hip -> L_Knee -> L_Ankle
        (2, 4), (4, 6),      # R_Hip -> R_Knee -> R_Ankle
        (0, 7),              # Pelvis -> Neck
        (7, 8), (7, 9),      # Neck -> L_Shoulder, R_Shoulder
        (8, 10), (10, 12),   # L_Shoulder -> L_Elbow -> L_Wrist
        (9, 11), (11, 13),   # R_Shoulder -> R_Elbow -> R_Wrist
        (5, 14), (5, 15), (5, 16),  # L_Ankle -> toes/heel
        (6, 17), (6, 18), (6, 19),  # R_Ankle -> toes/heel
        (7, 24),             # Neck -> Nose
        (24, 22), (24, 23),  # Nose -> Eyes
        (22, 20), (23, 21),  # Eyes -> Ears
    ]

    lhand_skeleton = [
        (12, 25), (25, 26), (26, 27), (27, 28),  # Thumb
        (12, 29), (29, 30), (30, 31), (31, 32),  # Index
        (12, 33), (33, 34), (34, 35), (35, 36),  # Middle
        (12, 37), (37, 38), (38, 39), (39, 40),  # Ring
        (12, 41), (41, 42), (42, 43), (43, 44),  # Pinky
    ]

    rhand_skeleton = [
        (13, 45), (45, 46), (46, 47), (47, 48),  # Thumb
        (13, 49), (49, 50), (50, 51), (51, 52),  # Index
        (13, 53), (53, 54), (54, 55), (55, 56),  # Middle
        (13, 57), (57, 58), (58, 59), (59, 60),  # Ring
        (13, 61), (61, 62), (62, 63), (63, 64),  # Pinky
    ]

    face_contour_skeleton = [(i, i+1) for i in range(120, 136)]

    skeleton_parts = {
        'body': (body_skeleton, (0, 255, 255)),      # Yellow
        'lhand': (lhand_skeleton, (0, 255, 0)),      # Green
        'rhand': (rhand_skeleton, (255, 0, 0)),      # Blue
        'face': (face_contour_skeleton, (255, 0, 255)),  # Magenta
    }

    # 6. Draw bones first
    num_joints = joint_proj_scaled.shape[0]
    for part_name, (skeleton, color) in skeleton_parts.items():
        for (i, j) in skeleton:
            if i >= num_joints or j >= num_joints:
                continue
            x1, y1 = int(joint_proj_scaled[i, 0]), int(joint_proj_scaled[i, 1])
            x2, y2 = int(joint_proj_scaled[j, 0]), int(joint_proj_scaled[j, 1])
            # Check bounds
            if (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0] and
                0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]):
                cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 7. Draw keypoints
    for part_name, (indices, color) in body_parts.items():
        for idx in indices:
            if idx >= num_joints:
                continue
            x = int(joint_proj_scaled[idx, 0])
            y = int(joint_proj_scaled[idx, 1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                radius = 2
                cv2.circle(img, (x, y), radius, color, -1)

    cv2.imshow(window_name, img)
    key = cv2.waitKey(wait_key)

    if key == 27:
        cv2.destroyWindow(window_name)