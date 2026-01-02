import numpy as np
import os

# 1. Path setup
splits_dir = r'D:\Dev\Dataset\UBody\splits'

# 2. Process inter_scene
print('=== inter_scene ===')
inter_path = os.path.join(splits_dir, 'inter_scene_test_list.npy')
inter = np.load(inter_path)
print(f'Original count: {len(inter)}')

np.random.seed(42)
np.random.shuffle(inter)

mid = len(inter) // 2
inter_valid = np.sort(inter[:mid])
inter_test = np.sort(inter[mid:])

np.save(os.path.join(splits_dir, 'inter_scene_valid_list.npy'), inter_valid)
np.save(os.path.join(splits_dir, 'inter_scene_test_list.npy'), inter_test)
print(f'valid: {len(inter_valid)}, test: {len(inter_test)}')

# 3. Process intra_scene
print('\n=== intra_scene ===')
intra_path = os.path.join(splits_dir, 'intra_scene_test_list.npy')
intra = np.load(intra_path)
print(f'Original count: {len(intra)}')

np.random.seed(42)
np.random.shuffle(intra)

mid = len(intra) // 2
intra_valid = np.sort(intra[:mid])
intra_test = np.sort(intra[mid:])

np.save(os.path.join(splits_dir, 'intra_scene_valid_list.npy'), intra_valid)
np.save(os.path.join(splits_dir, 'intra_scene_test_list.npy'), intra_test)
print(f'valid: {len(intra_valid)}, test: {len(intra_test)}')

# 4. Print result
print('\n=== Done ===')
print(f'Save path: {splits_dir}')