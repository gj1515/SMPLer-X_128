# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMPLer-X is a foundation model for expressive human pose and shape estimation from images/videos. It estimates SMPL-X body model parameters (body pose, hand pose, facial expression, shape) from input images using a ViT-based encoder with specialized heads for different body parts.

## Commands

### Installation
```bash
pip install -r requirements.txt

# Build mmcv with ops (required)
cd mmcv
set MMCV_WITH_OPS=1
pip install -v -e .
cd ..
```

### Inference (Video)
```bash
cd main
python inference.py \
    --pretrained_model smpler_x_h32 \
    --video_input path/to/video.mp4 \
    --video_output path/to/output.mp4 \
    --multi_person True \
    --save_mesh True
```

### Inference (Images)
```bash
cd main
python inference.py \
    --pretrained_model smpler_x_h32 \
    --img_path path/to/images \
    --output_folder path/to/output \
    --start 1 --end 100
```

### Training (DDP)
```bash
cd main
python train.py \
    --num_gpus 8 \
    --exp_name output/train_experiment \
    --master_port 45678 \
    --config config_smpler_x_h32.py
```

### Testing
```bash
cd main
python test.py \
    --num_gpus 1 \
    --exp_name output/test_experiment \
    --result_path train_output_dir \
    --ckpt_idx 9 \
    --testset EHF
```

### Mesh Overlay Visualization
```bash
cd main
python render.py \
    --data_path /path/to/inference/results \
    --seq video_name \
    --render_biggest_person True
```

## Architecture

### Model Pipeline (main/SMPLer_X.py)
1. **Encoder**: ViT backbone (S/B/L/H variants) pretrained on ViTPose, outputs image features + task tokens
2. **Body Position Net**: Predicts 3D heatmaps for 25 body joints
3. **Body Rotation Net**: Regresses root pose, body pose (21 joints), shape (10 betas), camera params from body tokens
4. **Box Net**: Predicts bounding boxes for hands and face from body features
5. **Hand RoI Net**: Differentiable feature crop-upsample using ROI Align for hands
6. **Hand Position/Rotation Net**: Per-hand joint prediction and pose regression (15 joints each)
7. **Face Regressor**: Predicts expression (10 params) and jaw pose from face tokens

### Key Design Patterns
- Task tokens from ViT encode different body part information (shape, camera, expression, jaw, hands, body pose)
- Hands are flipped to share weights (left hand is flipped to right hand coordinate frame)
- ROI-based hand processing allows higher resolution for detailed hand estimation
- All pose outputs use 6D rotation representation, converted to axis-angle for SMPL-X

### Directory Structure
- `main/`: Training, testing, inference entry points and configs
- `main/config/`: Model configs (e.g., `config_smpler_x_h32.py` for ViT-Huge with 32 datasets)
- `common/`: Shared utilities, network layers, body models
- `common/nets/`: Neural network components (PositionNet, RotationNet, etc.)
- `common/utils/human_model_files/`: SMPL/SMPL-X body model files
- `data/`: Dataset implementations (each dataset in its own folder)
- `main/transformer_utils/`: ViT encoder configs and mmpose integration

### Configuration System
- Uses mmcv Config for file-based configuration
- `main/config.py`: Runtime config manager that loads from config files
- Config files specify: model variant, datasets, training hyperparameters, input/output shapes
- Key config params: `encoder_config_file`, `feat_dim`, `trainset_3d`, `trainset_2d`, `trainset_humandata`

### Dataset Handling
- `data/dataset.py`: MultipleDatasets class for mixing multiple datasets with balancing strategies
- Each dataset class in `data/{DatasetName}/{DatasetName}.py`
- Supports HumanData format for unified annotation handling
- Data strategies: 'concat' (simple concatenation) or 'balance' (equal sampling per dataset)

### Human Body Models (common/utils/human_models.py)
- `smpl_x`: SMPL-X model wrapper with 137 joints (body + hands + face keypoints)
- Joint partitioning: body (0-24), lhand (25-44), rhand (45-64), face (65-136)
- pos_joint_num = 65 for PositionNet (excludes face keypoints)

### Person Detection
The codebase uses YOLO for person detection in inference (inference.py). Detection is done per-frame, and each detected person is processed independently through the model.

## Known Issues / FAQ

1. **torchgeometry '-' operator error**: Modify torchgeometry source to use `~` or `logical_not()` instead of `-` for bool tensors.

2. **Duplicate module registration KeyError**: Add `force=True` to module registration decorators in `main/transformer_utils/mmpose/models/utils/`.

## Model Variants
- SMPLer-X-S32: ViT-Small, 32M params, feat_dim=384
- SMPLer-X-B32: ViT-Base, 103M params, feat_dim=768
- SMPLer-X-L32: ViT-Large, 327M params, feat_dim=1024
- SMPLer-X-H32: ViT-Huge, 662M params, feat_dim=1280

The "32" suffix indicates training on 32 datasets. Number suffix can vary (5, 10, 20, 32).