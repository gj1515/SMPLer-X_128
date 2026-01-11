import os
import os.path as osp
import numpy as np
from humandata import HumanDataset
from config import cfg


class SignAvatar(HumanDataset):
    _cached_train_datalist = None  # Class-level cache for train data
    _cached_valid_datalist = None  # Class-level cache for valid data

    """
    Features:
    - SMPLX parameters: body, hands, face, expression, eyelid
    - 2D keypoints (SMPLX 144 format)
    - Resolution: 1280x720
    - Hand validity flags: left_valid, right_valid
    - Camera parameters: focal, princpt (per-frame, averaged for dataset)

    Data split (Fixed by SH Heo 251224):
    - Train: SignAvatar_train.npz (with interval + offset sampling)
    - Valid: SignAvatar_valid.npz (with valid_ratio random sampling, seed=42)
    - Test: SignAvatar_test.npz
    """

    def __init__(self, transform, data_split):
        super(SignAvatar, self).__init__(transform, data_split)

        self.img_dir = osp.join('C:/Users/user/Desktop/Dev/Dataset', 'SignAvatar', 'images')

        if data_split == 'train':
            self.annot_path = osp.join('C:/Users/user/Desktop/Dev/Dataset', 'SignAvatar', 'SignAvatar_train.npz')
        elif data_split == 'valid':
            self.annot_path = osp.join('C:/Users/user/Desktop/Dev/Dataset', 'SignAvatar', 'SignAvatar_valid.npz')
        else:
            self.annot_path = osp.join('C:/Users/user/Desktop/Dev/Dataset', 'SignAvatar', 'SignAvatar_test.npz')

        self.annot_path_cache = None
        self.use_cache = False

        self.img_shape = (720, 1280)  # (height, width)

        self.cam_param = self._load_cam_param()

        # Get config values
        self.train_sample_interval = getattr(cfg, 'SignAvatar_train_sample_interval', 1)
        self.test_sample_interval = getattr(cfg, 'SignAvatar_test_sample_interval', 1)
        self.valid_ratio = getattr(cfg, 'SignAvatar_valid_ratio', 0.1)

        if data_split == 'train':
            # Load train data with interval (use class-level cache)
            self.sample_interval = self.train_sample_interval
            if SignAvatar._cached_train_datalist is None:
                SignAvatar._cached_train_datalist = self.load_data(
                    train_sample_interval=1,
                    test_sample_interval=1
                )
                 # Truncate expression to 10 coefficients
                for data in SignAvatar._cached_train_datalist:
                    if data['smplx_param']['expr'] is not None:
                        data['smplx_param']['expr'] = data['smplx_param']['expr'][:10]
                print(f'[SignAvatar] Train data cached: {len(SignAvatar._cached_train_datalist)}')
            else:
                print(f'[SignAvatar] Using cached train data')

            total_samples = len(SignAvatar._cached_train_datalist)
            self.all_indices = np.arange(total_samples)
            sampled_indices = self.all_indices[::self.sample_interval]
            self.datalist = [SignAvatar._cached_train_datalist[i] for i in sampled_indices]

            print(f'[SignAvatar] Train total: {total_samples}, '
                  f'After interval({self.sample_interval}): {len(self.datalist)}')

        elif data_split == 'valid':
            # Load valid data with valid_ratio sampling (use class-level cache)
            self.sample_interval = 1
            if SignAvatar._cached_valid_datalist is None:
                SignAvatar._cached_valid_datalist = self.load_data(
                    train_sample_interval=1,
                    test_sample_interval=1
                )
                # Truncate expression to 10 coefficients
                for data in SignAvatar._cached_valid_datalist:
                    if data['smplx_param']['expr'] is not None:
                        data['smplx_param']['expr'] = data['smplx_param']['expr'][:10]
                print(f'[SignAvatar] Valid data cached: {len(SignAvatar._cached_valid_datalist)}')
            else:
                print(f'[SignAvatar] Using cached valid data')

            # Apply valid_ratio with fixed seed
            total_samples = len(SignAvatar._cached_valid_datalist)
            num_samples = int(total_samples * self.valid_ratio)

            np.random.seed(42)
            sampled_indices = np.random.choice(total_samples, num_samples, replace=False)
            sampled_indices = np.sort(sampled_indices)

            self.datalist = [SignAvatar._cached_valid_datalist[i] for i in sampled_indices]
            print(f'[SignAvatar] Valid total: {total_samples}, '
                  f'After valid_ratio({self.valid_ratio}): {len(self.datalist)}')

        else:  # test
            self.sample_interval = self.test_sample_interval
            self.datalist = self.load_data(
                train_sample_interval=1,
                test_sample_interval=self.test_sample_interval
            )
            # Truncate expression to 10 coefficients
            for data in self.datalist:
                if data['smplx_param']['expr'] is not None:
                    data['smplx_param']['expr'] = data['smplx_param']['expr'][:10]
            total_samples = len(self.datalist)

        # Dataset info for print_dataset_info()
        sampled_samples = len(self.datalist)
        self.dataset_info = {
            'name': self.__class__.__name__,
            'original': total_samples if data_split != 'train' else len(SignAvatar._cached_train_datalist) if SignAvatar._cached_train_datalist else sampled_samples,
            'sampled': sampled_samples,
            'final': sampled_samples,
            'sample_interval': self.sample_interval,
        }

        print(f'[SignAvatar] Loaded {len(self.datalist)} samples for {data_split}')

    def set_epoch(self, epoch):
        """Placeholder for epoch-based data reloading (not used)."""
        pass

    def _load_cam_param(self):
        """Load per-frame camera parameters from NPZ file."""
        try:
            content = np.load(self.annot_path, allow_pickle=True)
            if 'meta' in content:
                meta = content['meta'].item()
                if 'focal' in meta and 'princpt' in meta:
                    self.all_focal = np.array(meta['focal'])
                    self.all_princpt = np.array(meta['princpt'])
                    self.cam_param_dict = {
                        'focal': self.all_focal,
                        'princpt': self.all_princpt
                    }
                    return {
                        'focal': tuple(self.all_focal[0].tolist()),
                        'princpt': tuple(self.all_princpt[0].tolist())
                    }
        except Exception as e:
            print(f'[SignAvatar] Failed to load camera params: {e}')
        self.all_focal = None
        self.all_princpt = None
        self.cam_param_dict = None
        return None

    def _get_cam_param_for_idx(self, original_idx):
        """Get camera parameters for a specific frame index."""
        if self.all_focal is not None and self.all_princpt is not None:
            return {
                'focal': tuple(self.all_focal[original_idx].tolist()),
                'princpt': tuple(self.all_princpt[original_idx].tolist())
            }
        return self.cam_param

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # Get original frame index from datalist
        data = self.datalist[idx]
        original_idx = data.get('idx', idx)

        # Set per-frame camera param
        self.cam_param = self._get_cam_param_for_idx(original_idx)

        return super().__getitem__(idx)