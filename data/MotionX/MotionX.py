import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import torch
from tqdm import tqdm
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
import random

class MotionX(torch.utils.data.Dataset):
    _dataset_info = {}  # Class variable for dataset info

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        # Sampling parameters
        self.train_sample_interval = getattr(cfg, 'MotionX_train_sample_interval', 1)
        self.valid_sample_interval = getattr(cfg, 'MotionX_valid_sample_interval', 1)
        self.offset_step = getattr(cfg, 'MotionX_train_offset', 1)

        self.img_path = osp.join(cfg.data_dir, 'Motion-X++', 'images')
        self.annot_path = osp.join(cfg.data_dir, 'Motion-X++', 'annotations')

        # mscoco(wholebody) joint set
        self.joint_set = {
            'joint_num': 134,  # body 24 (23 + pelvis), lhand 21, rhand 21, face 68
            'joints_name': \
                (
                'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe',
                'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # body part
                'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2',
                'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1',
                'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand
                'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2',
                'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1',
                'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand
                *['Face_' + str(i) for i in range(56, 73)],  # face contour
                *['Face_' + str(i) for i in range(5, 56)]  # face
                ),
            'flip_pairs': \
                ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 21), (19, 22), (20, 23),
                 # body part
                 (24, 45), (25, 46), (26, 47), (27, 48), (28, 49), (29, 50), (30, 51), (31, 52), (32, 53), (33, 54),
                 (34, 55), (35, 56), (36, 57), (37, 58), (38, 59), (39, 60), (40, 61), (41, 62), (42, 63), (43, 64),
                 (44, 65),  # hand part
                 (66, 82), (67, 81), (68, 80), (69, 79), (70, 78), (71, 77), (72, 76), (73, 75),  # face contour
                 (83, 92), (84, 91), (85, 90), (86, 89), (87, 88),  # face eyebrow
                 (97, 101), (98, 100),  # face below nose
                 (102, 111), (103, 110), (104, 109), (105, 108), (106, 113), (107, 112),  # face eyes
                 (114, 120), (115, 119), (116, 118), (121, 125), (122, 124),  # face mouth
                 (126, 130), (127, 129), (131, 133)  # face lip
                 )
        }

        # Load clip list from split file (train.txt / valid.txt / test.txt)
        # Format: {category}/{clip_name} per line (e.g., "animation/Ways_to_Catch_360_clip1")
        split_file = osp.join(self.annot_path, f'{data_split}.txt')
        if osp.exists(split_file):
            with open(split_file, 'r') as f:
                self.clip_list = [line.strip() for line in f if line.strip()]
            print(f'[MotionX] Loaded {len(self.clip_list)} clips from {split_file}')
        else:
            self.clip_list = []
            print(f'[MotionX] Split file not found: {split_file}')

        # Load data for this split
        self.datalist = self.load_data()

        # Apply interval (train/valid)
        if data_split == 'train' and self.train_sample_interval > 1:
            self.full_datalist = self.datalist
            self.sample_interval = self.train_sample_interval
            self.datalist = self.full_datalist[::self.sample_interval]
        elif data_split == 'valid' and self.valid_sample_interval > 1:
            self.full_datalist = self.datalist
            self.sample_interval = self.valid_sample_interval
            self.datalist = self.full_datalist[::self.sample_interval]
        else:
            self.full_datalist = self.datalist
            self.sample_interval = 1

        # Store dataset info (unified format for print_dataset_info)
        self.dataset_info = {
            'name': 'Motion-X++',
            'original': getattr(self, '_original_count', len(self.full_datalist)),
            'sampled': getattr(self, '_sampled_count', len(self.full_datalist)),
            'final': len(self.datalist),
            'sample_interval': self.sample_interval,
        }

    def set_epoch(self, epoch):
        """Reload data with new offset for cyclic sampling (train only)."""
        if self.data_split != 'train' or self.sample_interval <= 1:
            return

        offset = (epoch * self.offset_step) % self.sample_interval
        self.datalist = self.full_datalist[offset::self.sample_interval]

        # Update sampled count in dataset_info
        self.dataset_info['sampled'] = len(self.datalist)

    def merge_joint(self, joint_img, feet_img, lhand_img, rhand_img, face_img):
        # pelvis
        lhip_idx = self.joint_set['joints_name'].index('L_Hip')
        rhip_idx = self.joint_set['joints_name'].index('R_Hip')
        pelvis = (joint_img[lhip_idx, :] + joint_img[rhip_idx, :]) * 0.5
        pelvis[2] = joint_img[lhip_idx, 2] * joint_img[rhip_idx, 2]  # joint_valid
        pelvis = pelvis.reshape(1, 3)

        # feet
        lfoot = feet_img[:3, :]
        rfoot = feet_img[3:, :]

        joint_img = np.concatenate((joint_img, pelvis, lfoot, rfoot, lhand_img, rhand_img, face_img)).astype(
            np.float32)  # self.joint_set['joint_num'], 3
        return joint_img

    def load_data(self):
        """Load data from clip_list. Each clip has keypoints JSON and mesh_recovery JSON."""
        datalist = []
        original_count = 0
        sampled_count = 0
        skipped_clips = 0

        for clip_path in tqdm(self.clip_list, desc=f'[MotionX] Loading {self.data_split}'):
            # clip_path: "animation/Ways_to_Catch_360_clip1"
            category, clip_name = clip_path.split('/')

            # JSON paths
            kp_json_path = osp.join(self.annot_path, 'keypoints', category, f'{clip_name}.json')
            mesh_json_path = osp.join(self.annot_path, 'mesh_recovery', 'local_motion', category, f'{clip_name}.json')

            # Skip if JSON files don't exist
            if not osp.exists(kp_json_path) or not osp.exists(mesh_json_path):
                skipped_clips += 1
                continue

            # Load JSONs
            with open(kp_json_path, 'r') as f:
                kp_data = json.load(f)
            with open(mesh_json_path, 'r') as f:
                mesh_data = json.load(f)

            # Index mesh annotations by image_id
            mesh_ann_dict = {ann['image_id']: ann for ann in mesh_data['annotations']}

            # Process each frame
            for kp_ann in kp_data['annotations']:
                original_count += 1
                image_id = kp_ann['image_id']

                # Image path: images/{category}/{clip}/frame_{num}.jpg
                frame_num = kp_ann['file_name'].split('/')[-1].split('.')[0]
                img_path = osp.join(self.img_path, category, clip_name, f'frame_{frame_num}.jpg')
                # Skip if image not found
                if not osp.exists(img_path):
                    continue

                # Get corresponding mesh annotation
                mesh_ann = mesh_ann_dict.get(image_id)
                if mesh_ann is None:
                    continue

                # Skip if smplx_params contains NaN
                smplx_params = mesh_ann['smplx_params']
                has_nan = False
                for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'shape', 'expr', 'trans']:
                    if key in smplx_params and smplx_params[key] is not None:
                        if np.any(np.isnan(smplx_params[key])):
                            has_nan = True
                            break
                if has_nan:
                    continue

                # bbox from mesh_recovery [x, y, w, h]
                bbox = np.array(mesh_ann['bbox'], dtype=np.float32)
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                sampled_count += 1

                # Joint coordinates (Motion-X++ format)
                body_kpts = np.array(kp_ann['body_kpts'], dtype=np.float32).reshape(-1, 3)
                foot_kpts = np.array(kp_ann['foot_kpts'], dtype=np.float32).reshape(-1, 3)
                lhand_kpts = np.array(kp_ann['lefthand_kpts'], dtype=np.float32).reshape(-1, 3)
                rhand_kpts = np.array(kp_ann['righthand_kpts'], dtype=np.float32).reshape(-1, 3)
                face_kpts = np.array(kp_ann['face_kpts'], dtype=np.float32).reshape(-1, 3)

                joint_img = self.merge_joint(body_kpts, foot_kpts, lhand_kpts, rhand_kpts, face_kpts)
                joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                joint_img[:, 2] = 0

                # Fill hand/face annotation from body
                for body_name, part_name in (
                    ('L_Wrist', 'L_Wrist_Hand'), ('R_Wrist', 'R_Wrist_Hand'), ('Nose', 'Face_18')):
                    if joint_valid[self.joint_set['joints_name'].index(part_name), 0] == 0:
                        joint_img[self.joint_set['joints_name'].index(part_name)] = joint_img[
                            self.joint_set['joints_name'].index(body_name)]
                        joint_valid[self.joint_set['joints_name'].index(part_name)] = joint_valid[
                            self.joint_set['joints_name'].index(body_name)]

                # Compute hand/face bbox from keypoints
                lhand_bbox = self._compute_bbox_from_keypoints(lhand_kpts)
                rhand_bbox = self._compute_bbox_from_keypoints(rhand_kpts)
                face_bbox = self._compute_bbox_from_keypoints(face_kpts)

                # Convert smplx_params
                smplx_param = self._convert_smplx_params(mesh_ann, bbox)

                # Generate unique ann_id for evaluate()
                ann_id = f'{category}_{clip_name}_{image_id}'

                data_dict = {
                    'img_path': img_path,
                    'img_shape': None,  # Will be loaded in __getitem__
                    'bbox': bbox,
                    'ann_id': ann_id,
                    'joint_img': joint_img,
                    'joint_valid': joint_valid,
                    'smplx_param': smplx_param,
                    'lhand_bbox': lhand_bbox,
                    'rhand_bbox': rhand_bbox,
                    'face_bbox': face_bbox,
                }
                datalist.append(data_dict)

        # Store counts
        self._original_count = original_count
        self._sampled_count = sampled_count

        print(f'[MotionX] {self.data_split}: clips={len(self.clip_list)}, skipped={skipped_clips}, '
              f'original={original_count}, sampled={sampled_count}')

        if getattr(cfg, 'data_strategy', None) == 'balance':
            print(f"[MotionX] Using [balance] strategy with datalist shuffled...")
            random.shuffle(datalist)

        return datalist

    def _compute_bbox_from_keypoints(self, keypoints, conf_thresh=0.5, min_kpts=15, expand_ratio=1.5):
        """Compute bounding box from keypoints.

        Args:
            keypoints: (N, 3) array with x, y, confidence
            conf_thresh: confidence threshold
            min_kpts: minimum number of valid keypoints required
            expand_ratio: bbox expansion ratio (1.5 = 150% of original size)

        Returns:
            bbox: [xmin, ymin, xmax, ymax] or None if not enough valid keypoints
        """
        # Filter valid keypoints
        valid_mask = keypoints[:, 2] > conf_thresh
        valid_kpts = keypoints[valid_mask]

        if len(valid_kpts) < min_kpts:
            return None

        # Compute bbox
        xmin, ymin = valid_kpts[:, 0].min(), valid_kpts[:, 1].min()
        xmax, ymax = valid_kpts[:, 0].max(), valid_kpts[:, 1].max()

        # Expand bbox from center
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        w, h = xmax - xmin, ymax - ymin
        new_w, new_h = w * expand_ratio, h * expand_ratio

        xmin = cx - new_w / 2
        ymin = cy - new_h / 2
        xmax = cx + new_w / 2
        ymax = cy + new_h / 2

        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

    def _convert_smplx_params(self, mesh_ann, bbox):
        """Convert Motion-X++ smplx_params to expected format.

        Motion-X++: smplx_params, camera_params (not used, use cfg instead)
        Expected: smplx_param, cam_param (focal, princpt scaled by bbox)
        """
        smplx_params = mesh_ann['smplx_params']

        # Use cfg focal/princpt scaled by bbox (same as MSCOCO evaluate)
        focal = [cfg.focal[0], cfg.focal[1]]
        princpt = [cfg.princpt[0], cfg.princpt[1]]

        focal[0] = focal[0] / cfg.input_body_shape[1] * bbox[2]
        focal[1] = focal[1] / cfg.input_body_shape[0] * bbox[3]
        princpt[0] = princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0]
        princpt[1] = princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]

        return {
            'smplx_param': {
                'root_pose': smplx_params['root_pose'],
                'body_pose': smplx_params['body_pose'],
                'lhand_pose': smplx_params['lhand_pose'],
                'rhand_pose': smplx_params['rhand_pose'],
                'jaw_pose': smplx_params['jaw_pose'],
                'shape': smplx_params['shape'],
                'expr': smplx_params['expr'],
                'trans': smplx_params['trans'],
                'lhand_valid': True,  # TODO: compute from keypoints
                'rhand_valid': True,
                'face_valid': True,
            },
            'cam_param': {
                'focal': focal,
                'princpt': princpt,
            }
        }

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0]);
            xmax = np.max(bbox[:, 0]);
            ymin = np.min(bbox[:, 1]);
            ymax = np.max(bbox[:, 1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        # train/valid mode
        if self.data_split in ['train', 'valid']:
            img_path = data['img_path']

            # image load
            img = load_img(img_path)
            img_shape = (img.shape[0], img.shape[1])  # (height, width) from loaded image
            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape, img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.;
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.;
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0];
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
            face_bbox_size = face_bbox[1] - face_bbox[0];

            # coco gt
            dummy_coord = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)
            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], np.zeros_like(joint_img[:, :1])), 1)  # x, y, dummy depth
            joint_img, joint_cam, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(joint_img, dummy_coord,
                                                                              data['joint_valid'], do_flip, img_shape,
                                                                              self.joint_set['flip_pairs'],
                                                                              img2bb_trans, rot,
                                                                              self.joint_set['joints_name'],
                                                                              smpl_x.joints_name)

            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            if smplx_param is not None:
                smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig \
                    = process_human_model_output(smplx_param['smplx_param'], smplx_param['cam_param'], do_flip,
                                                 img_shape, img2bb_trans, rot, 'smplx')
                is_valid_fit = True

            else:
                # dummy values
                smplx_joint_img = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_cam = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_trunc = np.zeros((smpl_x.joint_num, 1), dtype=np.float32)
                smplx_joint_valid = np.zeros((smpl_x.joint_num), dtype=np.float32)
                smplx_pose = np.zeros((smpl_x.orig_joint_num * 3), dtype=np.float32)
                smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
                smplx_expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
                smplx_pose_valid = np.zeros((smpl_x.orig_joint_num), dtype=np.float32)
                smplx_expr_valid = False
                is_valid_fit = False

            # SMPLX pose parameter validity
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc

            # make zero mask for invalid fit
            if not is_valid_fit:
                smplx_pose_valid[:] = 0
                smplx_joint_valid[:] = 0
                smplx_joint_trunc[:] = 0
                smplx_shape_valid = False
            else:
                smplx_shape_valid = True

            inputs = {'img': img}
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smplx_joint_img': smplx_joint_img,
                       'smplx_joint_cam': smplx_joint_cam,
                       'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr,
                       'lhand_bbox_center': lhand_bbox_center,
                       'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center,
                       'rhand_bbox_size': rhand_bbox_size,
                       'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid,
                         'smplx_joint_trunc': smplx_joint_trunc,
                         'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': float(smplx_shape_valid),
                         'smplx_expr_valid': float(smplx_expr_valid), 'is_3D': float(False),
                        #  'lhand_bbox_valid': float(False), 'rhand_bbox_valid': float(False),
                        # 'face_bbox_valid': float(False)}
                         'lhand_bbox_valid': lhand_bbox_valid,
                         'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid}
            return inputs, targets, meta_info

        # test mode
        else:
            img_path = data['img_path']

            # image load
            img = load_img(img_path)
            img_shape = (img.shape[0], img.shape[1])  # (height, width) from loaded image
            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.

            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['ann_id']
            out = outs[n]

            if annot['lhand_bbox'] is not None:
                lhand_bbox = out['lhand_bbox'].copy().reshape(2, 2)
                lhand_bbox = np.concatenate((lhand_bbox, np.ones((2, 1))), 1)
                lhand_bbox = np.dot(out['bb2img_trans'], lhand_bbox.transpose(1, 0)).transpose(1, 0)[:, :2]

            if annot['rhand_bbox'] is not None:
                rhand_bbox = out['rhand_bbox'].copy().reshape(2, 2)
                rhand_bbox = np.concatenate((rhand_bbox, np.ones((2, 1))), 1)
                rhand_bbox = np.dot(out['bb2img_trans'], rhand_bbox.transpose(1, 0)).transpose(1, 0)[:, :2]

            if annot['face_bbox'] is not None:
                face_bbox = out['face_bbox'].copy().reshape(2, 2)
                face_bbox = np.concatenate((face_bbox, np.ones((2, 1))), 1)
                face_bbox = np.dot(out['bb2img_trans'], face_bbox.transpose(1, 0)).transpose(1, 0)[:, :2]

            vis = False
            if vis:
                img_path = annot['img_path']

                bbox = annot['bbox']
                focal = list(cfg.focal)
                princpt = list(cfg.princpt)
                focal[0] = focal[0] / cfg.input_body_shape[1] * bbox[2]
                focal[1] = focal[1] / cfg.input_body_shape[0] * bbox[3]
                princpt[0] = princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0]
                princpt[1] = princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]
                param_save = {'smplx_param': {'root_pose': out['smplx_root_pose'].tolist(),
                                              'body_pose': out['smplx_body_pose'].tolist(),
                                              'lhand_pose': out['smplx_lhand_pose'].tolist(),
                                              'rhand_pose': out['smplx_rhand_pose'].tolist(),
                                              'jaw_pose': out['smplx_jaw_pose'].tolist(),
                                              'shape': out['smplx_shape'].tolist(), 'expr': out['smplx_expr'].tolist(),
                                              'trans': out['cam_trans'].tolist()},
                              'cam_param': {'focal': focal, 'princpt': princpt}
                              }
                with open(str(ann_id) + '.json', 'w') as f:
                    json.dump(param_save, f)

        return {}

    def print_eval_result(self, eval_result):
        return
