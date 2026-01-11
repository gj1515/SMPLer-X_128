import os
import os.path as osp
import numpy as np
from humandata import HumanDataset
from config import cfg


class MotionX(HumanDataset):
    """
    Motion-X++ dataset loader for HumanData NPZ format.

    Expected NPZ structure (from convert_motionx_to_humandata.py):
        - image_path: (N,) str array
        - bbox_xywh: (N, 5) float array
        - keypoints2d: (N, 131, 3) compressed keypoints in HUMAN_DATA format
        - keypoints2d_mask: (190,) mask for decompression
        - smplx: dict with global_orient, body_pose, left_hand_pose, etc.
        - meta: dict with height, width, focal_length, principal_point
        - __keypoints_compressed__: True

    Uses separate NPZ files for train/valid/test splits.
    Uses parent class HumanDataset.load_data() for all processing.
    """

    def __init__(self, transform, data_split):
        super(MotionX, self).__init__(transform, data_split)

        # Path setup
        self.img_dir = osp.join(cfg.data_dir, 'Motion-X++', 'images')
        annot_dir = osp.join(cfg.data_dir, 'Motion-X++', 'annotations')

        # NPZ file selection by split
        if data_split == 'train':
            self.annot_path = osp.join(annot_dir, 'train.npz')
        elif data_split == 'valid':
            self.annot_path = osp.join(annot_dir, 'valid.npz')
        else:  # test
            self.annot_path = osp.join(annot_dir, 'test.npz')

        self.annot_path_cache = osp.join(cfg.data_dir, 'Motion-X++', 'cache', f'motionx_{data_split}.npz')

        # Use bbox-based camera parameters (no fixed cam_param)
        # Set to {} instead of None to avoid TypeError in process_human_model_output
        self.cam_param = {}

        # Sampling parameters
        self.train_sample_interval = getattr(cfg, 'MotionX_train_sample_interval', 1)
        self.valid_sample_interval = getattr(cfg, 'MotionX_valid_sample_interval', 1)

        self.use_cache = getattr(cfg, 'use_cache', False)

        # Get original size from NPZ for dataset_info
        content = np.load(self.annot_path, allow_pickle=True)
        original_samples = len(content['image_path'])
        original_images = len(np.unique(content['image_path']))
        content.close()

        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] Loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            # Use parent class load_data() - handles keypoints decompression,
            # SMPLX_137_MAPPING, SMPLX key conversion, etc.
            if data_split == 'train':
                self.datalist = self.load_data(
                    train_sample_interval=self.train_sample_interval,
                    test_sample_interval=1
                )
            elif data_split == 'valid':
                self.datalist = self.load_data(
                    train_sample_interval=1,
                    test_sample_interval=1
                )
                # Apply valid_sample_interval
                if self.valid_sample_interval > 1:
                    self.datalist = self.datalist[::self.valid_sample_interval]
            else:  # test
                self.datalist = self.load_data(
                    train_sample_interval=1,
                    test_sample_interval=1
                )

            if self.use_cache:
                cache_dir = osp.dirname(self.annot_path_cache)
                if not osp.exists(cache_dir):
                    os.makedirs(cache_dir)
                self.save_cache(self.annot_path_cache, self.datalist)

        # Dataset info for print_dataset_info()
        sampled_samples = len(self.datalist)

        if data_split == 'train':
            sample_interval = self.train_sample_interval
        else:
            sample_interval = self.valid_sample_interval

        self.dataset_info = {
            'name': self.__class__.__name__,
            'original': original_samples,
            'sampled': sampled_samples,
            'final': sampled_samples,
            'sample_interval': sample_interval,
        }

    def set_epoch(self, epoch):
        """Placeholder for epoch-based data reloading (not used)."""
        pass