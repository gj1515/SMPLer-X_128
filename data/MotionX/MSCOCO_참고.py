import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
import random
from humandata import Cache

class MSCOCO(torch.utils.data.Dataset):
    _dataset_info = {}  # Class variable for dataset info

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        # Sampling parameters
        self.train_sample_interval = getattr(cfg, 'MSCOCO_train_sample_interval', 1)
        self.valid_sample_interval = getattr(cfg, 'MSCOCO_valid_sample_interval', 1)
        self.offset_step = getattr(cfg, 'MSCOCO_train_offset', 1)

        if os.path.exists(osp.join(cfg.data_dir, 'MSCOCO', 'img')):
            self.img_path = osp.join(cfg.data_dir, 'MSCOCO', 'img')
            self.annot_path = osp.join(cfg.data_dir, 'MSCOCO', 'annotations')
        else:
            self.img_path = osp.join(cfg.data_dir, 'coco')
            self.annot_path = osp.join(cfg.data_dir, 'coco', 'annotations')

        # mscoco joint set
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

        # Load image list from split file (train.txt / valid.txt / eval.txt)
        split_file = osp.join(self.annot_path, f'{data_split}.txt')
        if osp.exists(split_file):
            with open(split_file, 'r') as f:
                self.split_images = set(line.strip() for line in f)
            print(f'[MSCOCO] Loaded {len(self.split_images)} images from {split_file}')
        else:
            self.split_images = None
            print(f'[MSCOCO] Split file not found: {split_file}, loading all images')

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
            'name': 'MSCOCO',
            'original': getattr(self, '_original_count', len(self.full_datalist)),
            'sampled': getattr(self, '_sampled_count', len(self.full_datalist)),
            'final': len(self.datalist),
            'sample_interval': self.sample_interval,
        }

        print(f"[MSCOCO {data_split}] original {self.dataset_info['original']}, "
              f"sampled {self.dataset_info['sampled']}, "
              f"interval {self.sample_interval}, final {self.dataset_info['final']}")

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
        # train/valid use train annotation, test uses val annotation
        if self.data_split in ['train', 'valid']:
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_train_v1.0.json'))
            smplx_json_path = osp.join(self.annot_path, 'MSCOCO_train_SMPLX_all_NeuralAnnot.json')
            with open(smplx_json_path) as f:
                print(f'load SMPLX parameters from {smplx_json_path}')
                smplx_params = json.load(f)
        else:
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_val_v1.0.json'))

        # train/valid mode - load data filtered by split_images
        if self.data_split in ['train', 'valid']:
            datalist = []
            original_count = 0  # annotations in current split (before filtering)
            sampled_count = 0   # passed quality filtering (before interval)

            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]

                # Filter by split images (image-level split)
                if self.split_images is not None:
                    if img['file_name'] not in self.split_images:
                        continue

                # Count original annotations in this split
                original_count += 1

                imgname = osp.join('train2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)

                # Fixed by SH Heo (260108) - skip if image not found
                if not osp.exists(img_path):
                    continue

                # exclude the samples that are crowd or have few visible keypoints
                if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue

                # bbox
                bbox = process_bbox(ann['bbox'], img['width'], img['height'], ratio=getattr(cfg, 'bbox_ratio', 1.25))
                if bbox is None:
                    continue

                # Quality filtering passed
                sampled_count += 1

                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                foot_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1, 3)
                lhand_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1, 3)
                rhand_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1, 3)
                face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
                joint_img = self.merge_joint(joint_img, foot_img, lhand_img, rhand_img, face_img)

                joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                joint_img[:, 2] = 0

                # use body annotation to fill hand/face annotation
                for body_name, part_name in (
                ('L_Wrist', 'L_Wrist_Hand'), ('R_Wrist', 'R_Wrist_Hand'), ('Nose', 'Face_18')):
                    if joint_valid[self.joint_set['joints_name'].index(part_name), 0] == 0:
                        joint_img[self.joint_set['joints_name'].index(part_name)] = joint_img[
                            self.joint_set['joints_name'].index(body_name)]
                        joint_valid[self.joint_set['joints_name'].index(part_name)] = joint_valid[
                            self.joint_set['joints_name'].index(body_name)]

                # hand/face bbox
                if ann['lefthand_valid']:
                    lhand_bbox = np.array(ann['lefthand_box']).reshape(4)
                    if hasattr(cfg, 'bbox_ratio'):
                        lhand_bbox = process_bbox(lhand_bbox, img['width'], img['height'], ratio=cfg.bbox_ratio)
                    if lhand_bbox is not None:
                        lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
                else:
                    lhand_bbox = None
                if ann['righthand_valid']:
                    rhand_bbox = np.array(ann['righthand_box']).reshape(4)
                    if hasattr(cfg, 'bbox_ratio'):
                        rhand_bbox = process_bbox(rhand_bbox, img['width'], img['height'], ratio=cfg.bbox_ratio)
                    if rhand_bbox is not None:
                        rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
                else:
                    rhand_bbox = None
                if ann['face_valid']:
                    face_bbox = np.array(ann['face_box']).reshape(4)
                    if hasattr(cfg, 'bbox_ratio'):
                        face_bbox = process_bbox(face_bbox, img['width'], img['height'], ratio=cfg.bbox_ratio)
                    if face_bbox is not None:
                        face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
                else:
                    face_bbox = None

                if str(aid) in smplx_params:
                    smplx_param = smplx_params[str(aid)]
                    if 'lhand_valid' not in smplx_param['smplx_param']:
                        smplx_param['smplx_param']['lhand_valid'] = ann['lefthand_valid']
                        smplx_param['smplx_param']['rhand_valid'] = ann['righthand_valid']
                        smplx_param['smplx_param']['face_valid'] = ann['face_valid']
                else:
                    smplx_param = None

                data_dict = {'img_path': img_path, 'img_shape': (img['height'], img['width']), 'bbox': bbox,
                             'joint_img': joint_img, 'joint_valid': joint_valid, 'smplx_param': smplx_param,
                             'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox}
                datalist.append(data_dict)

            # Store counts for dataset_info
            self._original_count = original_count
            self._sampled_count = sampled_count

            print(f'[MSCOCO {self.data_split}] original: {original_count}, '
                  f'sampled: {sampled_count}, datalist: {len(datalist)}')

            if getattr(cfg, 'data_strategy', None) == 'balance':
                print(f"[MSCOCO] Using [balance] strategy with datalist shuffled...")
                random.shuffle(datalist)

            return datalist

        # test mode
        else:
            datalist = []
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('val2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)

                # bbox
                bbox = process_bbox(ann['bbox'], img['width'], img['height'], ratio=getattr(cfg, 'bbox_ratio', 1.25))
                if bbox is None: continue

                # hand/face bbox
                if ann['lefthand_valid']:
                    lhand_bbox = np.array(ann['lefthand_box']).reshape(4)
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
                else:
                    lhand_bbox = None
                if ann['righthand_valid']:
                    rhand_bbox = np.array(ann['righthand_box']).reshape(4)
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
                else:
                    rhand_bbox = None
                if ann['face_valid']:
                    face_bbox = np.array(ann['face_box']).reshape(4)
                    face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
                else:
                    face_bbox = None

                data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'], img['width']),
                             'bbox': bbox, 'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox}
                datalist.append(data_dict)
            return datalist

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
            img_path, img_shape = data['img_path'], data['img_shape']

            # image load
            img = load_img(img_path)
            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape,
                                                                     img2bb_trans)
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
            img_path, img_shape = data['img_path'], data['img_shape']

            # image load
            img = load_img(img_path)
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

                """
                img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                joint_img = out['joint_img'].copy()
                joint_img[:,0] = joint_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                joint_img[:,1] = joint_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                for j in range(len(joint_img)):
                    if j in smpl_x.pos_joint_part['body']:
                        cv2.circle(img, (int(joint_img[j][0]), int(joint_img[j][1])), 3, (0,0,255), -1)
                lhand_bbox = out['lhand_bbox'].reshape(2,2).copy()
                cv2.rectangle(img, (int(lhand_bbox[0][0]), int(lhand_bbox[0][1])), (int(lhand_bbox[1][0]), int(lhand_bbox[1][1])), (255,0,0), 3)
                rhand_bbox = out['rhand_bbox'].reshape(2,2).copy()
                cv2.rectangle(img, (int(rhand_bbox[0][0]), int(rhand_bbox[0][1])), (int(rhand_bbox[1][0]), int(rhand_bbox[1][1])), (255,0,0), 3)
                face_bbox = out['face_bbox'].reshape(2,2).copy()
                cv2.rectangle(img, (int(face_bbox[0][0]), int(face_bbox[0][1])), (int(face_bbox[1][0]), int(face_bbox[1][1])), (255,0,0), 3)
                cv2.imwrite(str(ann_id) + '.jpg', img)
                """

                # save_obj(out['smplx_mesh_cam'], smpl_x.face, img_id + '_' + str(ann_id) + '.obj')

                """
                img = load_img(img_path)[:,:,::-1]
                bbox = annot['bbox']
                focal = list(cfg.focal)
                princpt = list(cfg.princpt)
                focal[0] = focal[0] / cfg.input_body_shape[1] * bbox[2]
                focal[1] = focal[1] / cfg.input_body_shape[0] * bbox[3]
                princpt[0] = princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0]
                princpt[1] = princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]
                img = render_mesh(img, out['smplx_mesh_cam'], smpl_x.face, {'focal': focal, 'princpt': princpt})
                #img = cv2.resize(img, (512,512))
                cv2.imwrite(img_id + '_' + str(ann_id) + '.jpg', img)
                """

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

    def visualize_sample(self, idx, show_keypoints=True, show_mesh=True, window_name=None):
        """Visualize a sample from the dataset by overlaying SMPLX mesh on the image.

        Args:
            idx (int): Index of the sample to visualize
            show_keypoints (bool): Whether to show 2D keypoints overlay
            show_mesh (bool): Whether to show SMPLX mesh overlay
            window_name (str): Name of the window to display. If None, uses dataset name

        Returns:
            vis_img (np.ndarray): Visualization image with overlays
        """
        from common.utils.vis import vis_keypoints, render_mesh_on_image

        if window_name is None:
            window_name = f'{self.__class__.__name__} Sample Visualization'

        # Get raw data without augmentation
        data = copy.deepcopy(self.datalist[idx])
        img_path = data['img_path']
        img_shape = data['img_shape']
        bbox = data['bbox']
        smplx_param = data.get('smplx_param', None)
        joint_img = data['joint_img']
        joint_valid = data['joint_valid']

        # Load original image (BGR for cv2.imshow and render_mesh_on_image)
        img = load_img(img_path, order='BGR')
        if img is None:
            print(f"Failed to load image: {img_path}")
            return None

        print(f"\n=== Visualizing Sample {idx} ===")
        print(f"Image: {img_path}")
        print(f"Image shape: {img.shape}")
        print(f"Bbox: {bbox}")
        print(f"Has SMPLX params: {smplx_param is not None}")

        vis_img = img.copy()

        # Visualize 2D keypoints
        if show_keypoints and joint_img is not None:
            print("Drawing keypoints...")
            valid_joints = joint_img[joint_valid.squeeze() > 0]
            if len(valid_joints) > 0:
                vis_img = vis_keypoints(vis_img, valid_joints, alpha=0.7, radius=3)

        # Visualize SMPLX mesh
        if show_mesh and smplx_param is not None:
            print("Generating SMPLX mesh...")
            try:
                # Prepare SMPLX parameters for forward pass
                with torch.no_grad():
                    # Get gender-specific layer
                    smplx_params_dict = smplx_param['smplx_param']
                    gender = smplx_params_dict.get('gender', 'neutral')
                    if isinstance(gender, np.ndarray):
                        gender = gender.item() if gender.size == 1 else 'neutral'
                    if gender not in ['male', 'female', 'neutral']:
                        gender = 'neutral'

                    smplx_layer = smpl_x.layer[gender]

                    # Convert parameters to torch tensors
                    body_pose = torch.FloatTensor(smplx_params_dict['body_pose']).reshape(1, -1)

                    root_pose = torch.FloatTensor(smplx_params_dict['root_pose']).reshape(1, 3) if smplx_params_dict['root_pose'] is not None else torch.zeros(1, 3)
                    shape = torch.FloatTensor(smplx_params_dict['shape']).reshape(1, -1) if smplx_params_dict['shape'] is not None else torch.zeros(1, 10)
                    trans = torch.FloatTensor(smplx_params_dict['trans']).reshape(1, 3) if smplx_params_dict['trans'] is not None else torch.zeros(1, 3)

                    lhand_pose = torch.FloatTensor(smplx_params_dict['lhand_pose']).reshape(1, -1) if smplx_params_dict.get('lhand_valid', False) else torch.zeros(1, 45)
                    rhand_pose = torch.FloatTensor(smplx_params_dict['rhand_pose']).reshape(1, -1) if smplx_params_dict.get('rhand_valid', False) else torch.zeros(1, 45)
                    expr = torch.FloatTensor(smplx_params_dict['expr']).reshape(1, -1) if smplx_params_dict.get('face_valid', False) and smplx_params_dict['expr'] is not None else torch.zeros(1, 10)

                    # Handle jaw_pose, leye_pose, reye_pose if present
                    jaw_pose = torch.FloatTensor(smplx_params_dict['jaw_pose']).reshape(1, 3) if 'jaw_pose' in smplx_params_dict and smplx_params_dict['jaw_pose'] is not None else torch.zeros(1, 3)
                    leye_pose = torch.FloatTensor(smplx_params_dict['leye_pose']).reshape(1, 3) if 'leye_pose' in smplx_params_dict and smplx_params_dict['leye_pose'] is not None else torch.zeros(1, 3)
                    reye_pose = torch.FloatTensor(smplx_params_dict['reye_pose']).reshape(1, 3) if 'reye_pose' in smplx_params_dict and smplx_params_dict['reye_pose'] is not None else torch.zeros(1, 3)

                    # Forward pass through SMPLX
                    output = smplx_layer(
                        betas=shape,
                        global_orient=root_pose,
                        body_pose=body_pose,
                        left_hand_pose=lhand_pose,
                        right_hand_pose=rhand_pose,
                        jaw_pose=jaw_pose,
                        leye_pose=leye_pose,
                        reye_pose=reye_pose,
                        expression=expr,
                        transl=trans
                    )

                    vertices = output.vertices[0].cpu().numpy()

                    # Camera parameters for projection from smplx_param
                    if 'cam_param' in smplx_param:
                        focal = list(smplx_param['cam_param']['focal'])
                        princpt = list(smplx_param['cam_param']['princpt'])
                        print(f"Using SMPLX annotation camera parameters")
                    else:
                        # Fallback to config-based camera parameters scaled by bbox
                        focal = [self.cfg.model.focal[0] / self.cfg.model.input_body_shape[1] * bbox[2],
                                 self.cfg.model.focal[1] / self.cfg.model.input_body_shape[0] * bbox[3]]
                        princpt = [self.cfg.model.princpt[0] / self.cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                                   self.cfg.model.princpt[1] / self.cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
                        print(f"Using config-based camera parameters (scaled by bbox)")

                    cam_param = {
                        'focal': focal,
                        'princpt': princpt
                    }

                    print(f"Vertices shape: {vertices.shape}")
                    print(f"Camera focal: {focal}")
                    print(f"Camera princpt: {princpt}")

                    # Render mesh on image
                    faces = smpl_x.face
                    vis_img = render_mesh_on_image(vis_img, vertices, faces, cam_param)

            except Exception as e:
                print(f"Error generating SMPLX mesh: {e}")
                import traceback
                traceback.print_exc()
        elif show_mesh and smplx_param is None:
            print("Skipping mesh visualization: No SMPLX parameters available for this sample")

        return vis_img
