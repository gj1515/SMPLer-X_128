"""
Convert UBody split npy files to txt files.

Usage:
    python npy_to_txt.py
"""

import numpy as np
import os
import os.path as osp


def convert_npy_to_txt(splits_dir):
    """
    Convert all npy files in splits_dir to txt files.
    Each line in txt file contains one entry from npy array.
    """

    if not osp.exists(splits_dir):
        print(f'Error: splits directory not found: {splits_dir}')
        return

    # Find all npy files
    npy_files = [f for f in os.listdir(splits_dir) if f.endswith('.npy')]

    if not npy_files:
        print(f'No npy files found in {splits_dir}')
        return

    print(f'Found {len(npy_files)} npy files in {splits_dir}')
    print('=' * 50)

    for npy_file in sorted(npy_files):
        npy_path = osp.join(splits_dir, npy_file)
        txt_file = npy_file.replace('.npy', '.txt')
        txt_path = osp.join(splits_dir, txt_file)

        # Load npy file
        data = np.load(npy_path)

        # Save to txt file
        with open(txt_path, 'w') as f:
            for item in data:
                f.write(str(item) + '\n')

        print(f'{npy_file} -> {txt_file} ({len(data)} entries)')

    print('=' * 50)
    print(f'Done. Saved to: {splits_dir}')


if __name__ == '__main__':
    splits_dir = r'D:\Dev\Dataset\UBody\splits'
    convert_npy_to_txt(splits_dir)