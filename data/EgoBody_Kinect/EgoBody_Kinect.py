import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D
from utils.transforms import world2cam, cam2pixel, rigid_align
from humandata import HumanDataset


class EgoBody_Kinect(HumanDataset):
    def __init__(self, transform, data_split):
        super(EgoBody_Kinect, self).__init__(transform, data_split)

        if self.data_split == 'train':
            filename = getattr(cfg, 'filename', 'egobody_kinect_train.npz')
        elif self.data_split == 'valid':
            filename = getattr(cfg, 'filename', 'egobody_kinect_valid.npz')
        else:  # test
            filename = getattr(cfg, 'filename', 'egobody_kinect_test.npz')

        self.use_betas_neutral = getattr(cfg, 'egobody_fix_betas', False)

        self.img_dir = osp.join(cfg.data_dir, 'EgoBody')
        self.annot_path = osp.join(cfg.data_dir, 'EgoBody', filename)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', filename)
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.img_shape = (1080, 1920)  # (h, w)

        # Kinect master RGB camera intrinsics (from kinect_cam_params/kinect_master/Color.json)
        self.cam_param = {
            'focal': [918.241638, 918.177368],
            'princpt': [958.487976, 551.059509]
        }

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data(
                train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_{self.data_split}_sample_interval', 1))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)

    def visualize_sample(self, idx, show_keypoints=True, show_mesh=True, window_name=None):
        """Visualize a sample from the dataset by overlaying SMPLX mesh on the image.

        Args:
            idx (int): Index of the sample to visualize
            show_keypoints (bool): Whether to show 2D keypoints overlay
            show_mesh (bool): Whether to show SMPLX mesh overlay
            window_name (str): Name of the window to display

        Returns:
            vis_img (np.ndarray): Visualization image with overlays
        """
        from utils.vis import render_mesh_on_image

        if window_name is None:
            window_name = f'{self.__class__.__name__} Sample Visualization'

        # Get raw data without augmentation
        data = copy.deepcopy(self.datalist[idx])
        img_path = data['img_path']
        img_shape = data.get('img_shape', self.img_shape)
        bbox = data['bbox']
        smplx_param = data.get('smplx_param', None)
        joint_img = data.get('joint_img', None)
        joint_valid = data.get('joint_valid', None)

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
        if show_keypoints and joint_img is not None and joint_valid is not None:
            print("Drawing keypoints...")
            for j in range(joint_img.shape[0]):
                if joint_valid[j] > 0.5:
                    x, y = int(joint_img[j, 0]), int(joint_img[j, 1])
                    if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                        # Color by joint type
                        if j < 25:
                            color = (0, 255, 0)  # body - green
                        elif j < 45:
                            color = (255, 0, 0)  # left hand - blue
                        elif j < 65:
                            color = (0, 0, 255)  # right hand - red
                        else:
                            color = (255, 255, 0)  # face - cyan
                        cv2.circle(vis_img, (x, y), 3, color, -1)

        # Visualize SMPLX mesh
        if show_mesh and smplx_param is not None:
            print("Generating SMPLX mesh...")
            try:
                with torch.no_grad():
                    # Get SMPLX layer (use neutral gender)
                    smplx_layer = smpl_x.layer['neutral']

                    # Convert parameters to torch tensors
                    root_pose = smplx_param.get('root_pose')
                    if root_pose is not None:
                        root_pose = torch.FloatTensor(root_pose).reshape(1, 3)
                    else:
                        root_pose = torch.zeros(1, 3)

                    body_pose = smplx_param.get('body_pose')
                    if body_pose is not None:
                        body_pose = torch.FloatTensor(body_pose).reshape(1, -1)
                    else:
                        body_pose = torch.zeros(1, 63)

                    shape = smplx_param.get('shape')
                    if shape is not None:
                        shape = torch.FloatTensor(shape).reshape(1, -1)[:, :10]
                    else:
                        shape = torch.zeros(1, 10)

                    trans = smplx_param.get('trans')
                    if trans is not None:
                        trans = torch.FloatTensor(trans).reshape(1, 3)
                    else:
                        trans = torch.zeros(1, 3)

                    lhand_pose = smplx_param.get('lhand_pose')
                    if lhand_pose is not None:
                        lhand_pose = torch.FloatTensor(lhand_pose).reshape(1, -1)
                    else:
                        lhand_pose = torch.zeros(1, 45)

                    rhand_pose = smplx_param.get('rhand_pose')
                    if rhand_pose is not None:
                        rhand_pose = torch.FloatTensor(rhand_pose).reshape(1, -1)
                    else:
                        rhand_pose = torch.zeros(1, 45)

                    expr = smplx_param.get('expr')
                    if expr is not None:
                        expr = torch.FloatTensor(expr).reshape(1, -1)[:, :10]
                    else:
                        expr = torch.zeros(1, 10)

                    jaw_pose = smplx_param.get('jaw_pose')
                    if jaw_pose is not None:
                        jaw_pose = torch.FloatTensor(jaw_pose).reshape(1, 3)
                    else:
                        jaw_pose = torch.zeros(1, 3)

                    # Forward pass through SMPLX
                    output = smplx_layer(
                        betas=shape,
                        global_orient=root_pose,
                        body_pose=body_pose,
                        left_hand_pose=lhand_pose,
                        right_hand_pose=rhand_pose,
                        jaw_pose=jaw_pose,
                        leye_pose=torch.zeros(1, 3),
                        reye_pose=torch.zeros(1, 3),
                        expression=expr,
                        transl=trans
                    )

                    vertices = output.vertices[0].cpu().numpy()

                    # Use Kinect camera intrinsics
                    cam_param = self.cam_param

                    print(f"Vertices shape: {vertices.shape}")
                    print(f"Camera focal: {cam_param['focal']}")
                    print(f"Camera princpt: {cam_param['princpt']}")

                    # Render mesh on image
                    faces = smpl_x.face
                    vis_img = render_mesh_on_image(vis_img, vertices, faces, cam_param)
                    print("Mesh rendered successfully!")

            except Exception as e:
                print(f"Error generating SMPLX mesh: {e}")
                import traceback
                traceback.print_exc()
        elif show_mesh and smplx_param is None:
            print("Skipping mesh visualization: No SMPLX parameters available")

        return vis_img