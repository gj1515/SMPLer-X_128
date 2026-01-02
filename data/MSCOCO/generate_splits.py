"""
Split MSCOCO train2017 images into train/valid/eval sets.

Split unit: Image-level (all annotations of the same image belong to the same split)
Ratio: train 90%, valid 5%, eval 5%

Usage:
    cd data/MSCOCO
    python generate_splits.py
"""

import os
import os.path as osp
import numpy as np
from pycocotools.coco import COCO


def generate_splits(annot_path, seed=42, train_ratio=0.95, valid_ratio=0.025):
    """
    Split COCO-WholeBody train annotation into train/valid/eval by image.

    Args:
        annot_path: Path to annotation files
        seed: Random seed for reproducibility
        train_ratio: Train split ratio (default 0.9)
        valid_ratio: Valid split ratio (default 0.05)
        # eval_ratio = 1 - train_ratio - valid_ratio = 0.05
    """

    # 1. Load COCO annotation
    annot_file = osp.join(annot_path, 'coco_wholebody_train_v1.0.json')
    print(f'Loading annotation from {annot_file}...')
    db = COCO(annot_file)

    # 2. Extract images with valid annotations only
    #    (iscrowd=False, num_keypoints > 0)
    image_files = set()
    valid_ann_count = 0

    for ann in db.anns.values():
        # Skip crowd annotations or annotations without keypoints
        if ann['iscrowd'] or ann['num_keypoints'] == 0:
            continue

        valid_ann_count += 1
        img_info = db.loadImgs(ann['image_id'])[0]
        image_files.add(img_info['file_name'])

    print(f'Total annotations: {len(db.anns)}')
    print(f'Valid annotations (not crowd, has keypoints): {valid_ann_count}')
    print(f'Unique images with valid annotations: {len(image_files)}')

    # 3. Sort and shuffle for reproducibility
    image_list = sorted(list(image_files))
    np.random.seed(seed)
    np.random.shuffle(image_list)

    # 4. Split by ratio
    total = len(image_list)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train_images = image_list[:train_end]
    valid_images = image_list[train_end:valid_end]
    eval_images = image_list[valid_end:]

    # 5. Count annotations per split
    train_set = set(train_images)
    valid_set = set(valid_images)
    eval_set = set(eval_images)

    train_ann_count = 0
    valid_ann_count = 0
    eval_ann_count = 0

    for ann in db.anns.values():
        if ann['iscrowd'] or ann['num_keypoints'] == 0:
            continue
        img_info = db.loadImgs(ann['image_id'])[0]
        file_name = img_info['file_name']

        if file_name in train_set:
            train_ann_count += 1
        elif file_name in valid_set:
            valid_ann_count += 1
        elif file_name in eval_set:
            eval_ann_count += 1

    # 6. Save to text files (sorted)
    splits = [
        ('train.txt', train_images),
        ('valid.txt', valid_images),
        ('eval.txt', eval_images)
    ]

    for filename, images in splits:
        filepath = osp.join(annot_path, filename)
        with open(filepath, 'w') as f:
            for img in sorted(images):
                f.write(img + '\n')
        print(f'Saved {filepath}')

    # 7. Print results
    print('\n' + '=' * 50)
    print('Split Results (Image-level)')
    print('=' * 50)
    print(f'{"Split":<10} {"Images":>10} {"Ratio":>10} {"Annotations":>15}')
    print('-' * 50)
    print(f'{"train":<10} {len(train_images):>10} {len(train_images)/total*100:>9.1f}% {train_ann_count:>15}')
    print(f'{"valid":<10} {len(valid_images):>10} {len(valid_images)/total*100:>9.1f}% {valid_ann_count:>15}')
    print(f'{"eval":<10} {len(eval_images):>10} {len(eval_images)/total*100:>9.1f}% {eval_ann_count:>15}')
    print('-' * 50)
    print(f'{"Total":<10} {total:>10} {"100.0%":>10} {train_ann_count + valid_ann_count + eval_ann_count:>15}')
    print('=' * 50)


if __name__ == '__main__':
    # Set annotations path relative to script location
    annot_path = 'D:/Dev/Dataset/MSCOCO/annotations'

    if not osp.exists(annot_path):
        print(f'Error: annotations path not found: {annot_path}')
        print('Please run this script from data/MSCOCO directory')
        exit(1)

    generate_splits(annot_path)