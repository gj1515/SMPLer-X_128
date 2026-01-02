"""
Visualize training data samples before training
Usage: python visualize_training_data.py --num_samples 3
"""

import argparse
import sys
import os
import torchvision.transforms as transforms
import cv2
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

import importlib

# Add paths for SMPLer-X imports
sys.path.insert(0, str(root_dir / 'main'))
sys.path.insert(0, str(root_dir / 'data'))
sys.path.insert(0, str(root_dir / 'common'))

from config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize training data samples')
    parser.add_argument('--config', type=str, default='config_smpler_x_s3.py',
                        help='Config file name in main/config/ directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize from each dataset (default: 10)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting sample index (default: 0)')
    parser.add_argument('--show_keypoints', action='store_true', default=True,
                        help='Show 2D keypoints overlay')
    parser.add_argument('--show_mesh', action='store_true', default=True,
                        help='Show SMPLX mesh overlay')
    parser.add_argument('--no_keypoints', action='store_true',
                        help='Disable keypoints visualization')
    parser.add_argument('--no_mesh', action='store_true',
                        help='Disable mesh visualization')

    args = parser.parse_args()

    # Handle negative flags
    if args.no_keypoints:
        args.show_keypoints = False
    if args.no_mesh:
        args.show_mesh = False

    return args


def dynamic_import(module_name, object_name):
    """Dynamically import a module and access a specific object."""
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def main():
    args = parse_args()

    # Load config
    config_path = os.path.join(root_dir, 'main', 'config', args.config)
    cfg.get_config_fromfile(config_path)

    print(f"\n{'='*70}")
    print(f"Training Data Visualization Tool")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Training datasets: {cfg.trainset_humandata}")
    print(f"Number of samples per dataset: {args.num_samples}")
    print(f"Starting index: {args.start_idx}")
    print(f"Show keypoints: {args.show_keypoints}")
    print(f"Show mesh: {args.show_mesh}")
    print(f"{'='*70}\n")

    # Define transform
    transform = transforms.ToTensor()

    print(f"Human model path: {cfg.human_model_path}\n")

    # Load each training dataset
    for dataset_name in cfg.trainset_humandata:
        print(f"\n{'='*70}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*70}")

        try:
            # Dynamically import dataset from data/{dataset_name}/{dataset_name}.py
            dataset_class = dynamic_import(f"{dataset_name}.{dataset_name}", dataset_name)
            dataset = dataset_class(transform, 'train')

            print(f"Dataset loaded successfully!")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Dataset class: {dataset.__class__.__name__}")

            # Check if dataset has visualize_sample method
            if not hasattr(dataset, 'visualize_sample'):
                print(f"  [WARNING] Dataset {dataset_name} does not have visualize_sample method. Skipping.")
                continue

            # Visualize samples
            max_idx = min(args.start_idx + args.num_samples, len(dataset))
            samples_to_show = max_idx - args.start_idx

            print(f"\nVisualizing {samples_to_show} samples (indices {args.start_idx} to {max_idx-1})...\n")

            for i in range(args.start_idx, max_idx):
                try:
                    print(f"\n{'-'*70}")
                    print(f"Sample {i - args.start_idx + 1}/{samples_to_show} (Dataset index: {i})")
                    print(f"{'-'*70}")

                    # Call dataset's own visualize_sample method
                    vis_img = dataset.visualize_sample(
                        idx=i,
                        show_keypoints=args.show_keypoints,
                        show_mesh=args.show_mesh
                    )

                    if vis_img is not None:
                        # Display the image
                        window_name = f'{dataset_name} - Sample {i}'
                        cv2.imshow(window_name, vis_img)
                        print(f"Displaying visualization. Press any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print(f"[ERROR] Failed to visualize sample {i}")

                except KeyboardInterrupt:
                    print(f"\n\nVisualization interrupted by user.")
                    print(f"Skipping remaining samples in {dataset_name}...")
                    break
                except Exception as e:
                    print(f"[ERROR] Error visualizing sample {i}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Ask user if they want to continue
                    try:
                        response = input("\nContinue to next sample? (y/n): ")
                        if response.lower() != 'y':
                            break
                    except:
                        break

        except Exception as e:
            print(f"[ERROR] Error loading dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()