import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from human_models.human_models import SMPLX
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D
from utils.transforms import world2cam, cam2pixel, rigid_align
from datasets.humandata import HumanDataset


class GTA_Human2(HumanDataset):
    """
    Data split process (Fixed by SH Heo 251225):
    1. Load all data from npz (no interval)
    2. Build sequence-to-indices mapping (seq_xxx folders)
    3. Split sequences into train/valid with fixed seed (valid_ratio)
    4. Train: apply train_sample_interval, Valid: apply valid_sample_interval
    5. set_epoch changes offset within train_indices only
    """
    _cached_all_datalist = None
    _dataset_info = {}  # Class variable for sharing info between train/valid instances

    def __init__(self, transform, data_split, cfg):
        super(GTA_Human2, self).__init__(transform, data_split, cfg)

        self.smpl_x = SMPLX.get_instance()
        self.cfg = cfg

        filename = 'gta_human2.npz'
        self.img_dir = osp.join(cfg.data.data_dir, 'GTA_Human2')
        self.annot_path = osp.join(cfg.data.data_dir, 'GTA_Human2', 'preprocessed', filename)
        self.annot_path_cache = osp.join(cfg.data.data_dir, 'GTA_Human2', 'cache', filename)
        self.use_cache = getattr(cfg.data, 'use_cache', False)
        self.img_shape = (1080, 1920)  # (h, w)
        self.cam_param = {
            'focal': (1158.0337, 1158.0337),  # (fx, fy)
            'princpt': (960, 540)  # (cx, cy)
        }

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_path = img_path.replace('\\', '/')
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # Sampling parameters
        self.train_sample_interval = getattr(cfg.data, f'{self.__class__.__name__}_train_sample_interval', 1)
        self.valid_sample_interval = getattr(cfg.data, f'{self.__class__.__name__}_valid_sample_interval', 1)
        self.offset_step = getattr(cfg.data, f'{self.__class__.__name__}_train_offset', 1)
        self.valid_ratio = getattr(cfg.data, f'{self.__class__.__name__}_valid_ratio', 0.1)

        if data_split in ['train', 'valid']:
            # Use class variable to avoid pickle copy in multiprocessing
            if GTA_Human2._cached_all_datalist is None:
                GTA_Human2._cached_all_datalist = self.load_data(
                    train_sample_interval=1,
                    offset=0
                )
            total_samples = len(GTA_Human2._cached_all_datalist)

            # Build sequence-to-indices mapping
            seq_to_indices = {}
            for idx, data in enumerate(GTA_Human2._cached_all_datalist):
                img_path = data['img_path']
                # Extract sequence ID: ".../GTA_Human2/images/seq_00007425/00000000.jpeg" -> "seq_00007425"
                parts = img_path.replace('\\', '/').split('/')
                # Find seq_xxx in path
                seq_id = None
                for part in parts:
                    if part.startswith('seq_'):
                        seq_id = part
                        break
                if seq_id is None:
                    seq_id = 'unknown'
                seq_to_indices.setdefault(seq_id, []).append(idx)

            # Split sequences with fixed seed
            np.random.seed(42)
            all_sequences = sorted(seq_to_indices.keys())
            np.random.shuffle(all_sequences)

            split_idx = int(len(all_sequences) * (1 - self.valid_ratio))
            train_sequences = all_sequences[:split_idx]
            valid_sequences = all_sequences[split_idx:]

            # Flatten sequence -> frame indices
            self.train_indices = np.array([idx for seq in train_sequences for idx in seq_to_indices[seq]])
            self.valid_indices = np.array([idx for seq in valid_sequences for idx in seq_to_indices[seq]])

            # Apply interval within split
            if data_split == 'train':
                self.split_indices = self.train_indices
                self.sample_interval = self.train_sample_interval
                sampled_indices = self.split_indices[::self.sample_interval]
            else:  # valid
                self.split_indices = self.valid_indices
                self.sample_interval = self.valid_sample_interval
                sampled_indices = self.split_indices[::self.sample_interval]

            self.datalist = [GTA_Human2._cached_all_datalist[i] for i in sampled_indices]

            # Store dataset info using class variable
            if data_split == 'train':
                GTA_Human2._dataset_info = {
                    'name': 'GTA_Human2',
                    'total': total_samples,
                    'total_sequences': len(all_sequences),
                    'train_sequences': len(train_sequences),
                    'valid_sequences': len(valid_sequences),
                    'train_frames': len(self.train_indices),
                    'valid_frames': len(self.valid_indices),
                    'train_interval': self.train_sample_interval,
                    'valid_interval': self.valid_sample_interval,
                    'train_offset': self.offset_step,
                    'valid_ratio': self.valid_ratio,
                    'train_sampled': len(self.datalist),
                    'valid_sampled': 0
                }
                print(f"[GTA_Human2] total: {total_samples}, total_sequences: {len(all_sequences)}")
                print(f"[GTA_Human2] train_sequences: {len(train_sequences)}, valid_sequences: {len(valid_sequences)}")
                print(f"[GTA_Human2] train_frames: {len(self.train_indices)}, valid_frames: {len(self.valid_indices)}")
                print(f"[GTA_Human2] train_sampled: {len(self.datalist)}")
            else:  # valid
                GTA_Human2._dataset_info.update({
                    'valid_sampled': len(self.datalist)
                })
                print(f"[GTA_Human2] valid_sampled: {len(self.datalist)}")

        else:  # test
            self.sample_interval = 1
            self.datalist = self.load_data(
                train_sample_interval=1,
                offset=0
            )
            GTA_Human2._dataset_info = {
                'name': 'GTA_Human2',
                'total': len(self.datalist),
                'split': data_split,
                'sampled': len(self.datalist)
            }

        # Store reference to class variable
        self.dataset_info = GTA_Human2._dataset_info

    def set_epoch(self, epoch):
        """Reload data with new offset for cyclic sampling (train only)."""
        if self.data_split != 'train' or self.sample_interval <= 1:
            return

        offset = (epoch * self.offset_step) % self.sample_interval
        sampled_indices = self.split_indices[offset::self.sample_interval]
        self.datalist = [GTA_Human2._cached_all_datalist[i] for i in sampled_indices]

        # Update sampled count in dataset_info
        self.dataset_info['sampled'] = len(self.datalist)