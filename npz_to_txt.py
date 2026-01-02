import numpy as np
import argparse
import os


def npz_to_txt(npz_path, output_path=None):
    """Convert npz file to txt file."""
    if output_path is None:
        output_path = os.path.splitext(npz_path)[0] + '.txt'

    data = np.load(npz_path, allow_pickle=True)

    with open(output_path, 'w') as f:
        f.write(f"NPZ File: {npz_path}\n")
        f.write(f"Keys: {data.files}\n")
        f.write("=" * 80 + "\n\n")

        for key in data.files:
            f.write(f"[{key}]\n")
            f.write("-" * 40 + "\n")

            try:
                val = data[key]

                # Handle scalar array (dict wrapped in array)
                if val.shape == ():
                    item = val.item()
                    if isinstance(item, dict):
                        f.write(f"Type: dict\n")
                        f.write(f"Keys: {list(item.keys())}\n\n")
                        for k, v in item.items():
                            if hasattr(v, 'shape'):
                                f.write(f"  {k}: shape={v.shape}, dtype={v.dtype}\n")
                                # Write values if small enough
                                if v.size <= 100:
                                    f.write(f"  values: {v.flatten().tolist()}\n")
                            else:
                                f.write(f"  {k}: {type(v).__name__} = {v}\n")
                    else:
                        f.write(f"Type: {type(item).__name__}\n")
                        f.write(f"Value: {item}\n")
                else:
                    f.write(f"Shape: {val.shape}\n")
                    f.write(f"Dtype: {val.dtype}\n")

                    # Write values if small enough
                    if val.size <= 1000:
                        f.write(f"Values:\n{val}\n")
                    else:
                        f.write(f"Values (first 10 rows):\n{val[:10]}\n")
                        f.write(f"... ({val.shape[0] - 10} more rows)\n")

            except Exception as e:
                f.write(f"Error reading: {e}\n")

            f.write("\n")

    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, default=r"D:\Dev\Dataset\SynBody\synbody_v1_0\20230113\Downtown\LS_0114_004551_088_CAM002\smplx\people_0-Subject_15_F_6.npz")
    parser.add_argument('--output', type=str, default=r"D:\Dev\Dataset\SynBody\synbody_v1_0\20230113\Downtown\LS_0114_004551_088_CAM002\smplx\people_0-Subject_15_F_6.txt", help='Output txt path')
    args = parser.parse_args()

    npz_to_txt(args.npz_path, args.output)