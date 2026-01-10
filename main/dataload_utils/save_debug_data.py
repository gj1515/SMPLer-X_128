import json
import torch
import numpy as np
import os

def tensor_to_serializable(obj):
    """Convert tensors and numpy arrays to JSON-serializable format."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [tensor_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    else:
        return obj

def save_debug_data(targets, meta_info, save_dir='debug_output', prefix='sample'):
    """
    Save targets and meta_info to a single JSON file for debugging.

    Args:
        targets: dict of tensors
        meta_info: dict of tensors
        save_dir: output directory
        prefix: filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)

    # Combine targets and meta_info into single dict
    combined_data = {
        'targets': tensor_to_serializable(targets),
        'meta_info': tensor_to_serializable(meta_info)
    }

    # Save to single JSON file
    output_path = os.path.join(save_dir, f'{prefix}_debug_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    print(f"Saved debug data to: {output_path}")

    # Print summary of keys
    print("\n=== Targets Keys ===")
    for k, v in targets.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    print("\n=== Meta Info Keys ===")
    for k, v in meta_info.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    return output_path