"""
Script to convert pkl files in *_inference folders to npz format for C++ reading.

There are three types of pkl files expected in the *_inference folders:
- goal_inference/goal_reaching.pkl
    dict[str, np.ndarray] -> dict[str, np.ndarray]
- reward_inference/reward_prediction.pkl
    dict[str, list[torch.Tensor]] -> dict[str, np.ndarray],
    convert list of tensors to concatenated numpy arrays,
- tracking_inference/tracking.pkl
    np.ndarray -> dict[str, np.ndarray], {"data": np.ndarray}
Information about each pkl file will be written to the *.txt file in the same folder.

For each converted .npz file, a corresponding .txt file will be generated
containing the keys and shapes of the arrays in the npz file.

Usage:
    python scripts/latent_pkl2npz.py /home/yy/Coding/robot/BFM-Zero/model
"""
import argparse
import pickle
import joblib
import numpy as np
from pathlib import Path
from loguru import logger


def load_pkl_file(pkl_path: Path) -> dict:
    """
    Load a pkl file, trying different methods.

    Args:
        pkl_path: Path to the pkl file.

    Returns:
        The loaded data.
    """
    # Try joblib first (handles both joblib and pickle formats)
    try:
        data = joblib.load(pkl_path)
        logger.info(f"  Loaded with joblib: {pkl_path}")
        return data
    except Exception as e:
        logger.warning(f"  joblib.load failed for {pkl_path}: {e}")

    # Fallback to pickle
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"  Loaded with pickle: {pkl_path}")
        return data
    except Exception as e:
        logger.warning(f"  pickle.load failed for {pkl_path}: {e}")

    raise ValueError(f"Failed to load {pkl_path} with any method")


def check_pkl_format(pkl_path: Path) -> dict:
    """
    Check if the pkl file has the expected format: depth=1 dict with key->array.

    Args:
        pkl_path: Path to the pkl file.

    Returns:
        The loaded dictionary.

    Raises:
        ValueError: If the format is not as expected.
    """
    data = load_pkl_file(pkl_path)

    # If data is already a numpy array, wrap it in a dict
    if isinstance(data, np.ndarray):
        logger.info(f"  {pkl_path}: Raw numpy array, wrapping with key 'data'")
        return {"data": data}

    if not isinstance(data, dict):
        raise ValueError(f"{pkl_path}: Top level is not a dict, got {type(data)}")

    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"{pkl_path}: Key '{key}' is not a string, got {type(key)}")
        # Value can be np.ndarray, torch.Tensor, or list of these
        if isinstance(value, (np.ndarray, list)):
            pass  # Will be handled in convert_pkl_to_npz
        else:
            raise ValueError(f"{pkl_path}: Value for key '{key}' is not np.ndarray or list, got {type(value)}")

    return data


def convert_pkl_to_npz(pkl_path: Path, npz_path: Path) -> None:
    """
    Convert a pkl file to npz format.

    Args:
        pkl_path: Path to the input pkl file.
        npz_path: Path to the output npz file.
    """
    logger.info(f"Processing: {pkl_path}")

    # Check format
    data = check_pkl_format(pkl_path)

    # Convert values to numpy arrays for npz storage
    npz_data = {}
    array_info = []
    for key, value in data.items():
        if isinstance(value, list):
            # Convert list elements to numpy and concatenate
            np_list = []
            for item in value:
                if hasattr(item, 'cpu'):  # torch.Tensor
                    np_list.append(item.cpu().numpy())
                else:
                    np_list.append(np.asarray(item))
            npz_data[key] = np.concatenate(np_list, axis=0)
            array_info.append(f"{key}: {npz_data[key].shape}")
        else:
            if hasattr(value, 'cpu'):  # torch.Tensor
                npz_data[key] = value.cpu().numpy()
            else:
                npz_data[key] = np.asarray(value)
            array_info.append(f"{key}: {npz_data[key].shape}")

    logger.info(f"  Format OK: {len(data)} keys, arrays: {array_info}")

    # Save as npz (uncompressed for fast C++ loading)
    np.savez(npz_path, **npz_data)
    logger.info(f"  Saved: {npz_path}")

    # Save txt file with keys and shapes
    txt_path = npz_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(f"NPZ file: {npz_path.name}\n")
        f.write(f"Total keys: {len(npz_data)}\n\n")
        f.write("Keys and Shapes:\n")
        for key, value in npz_data.items():
            f.write(f"  {key}: {value.shape} (dtype: {value.dtype})\n")
    logger.info(f"  Saved: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pkl files to npz format for C++ reading"
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to the model folder containing *_inference subfolders"
    )
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    if not model_folder.exists():
        logger.error(f"Model folder does not exist: {model_folder}")
        return

    # Find all *_inference subfolders
    inference_folders = sorted(model_folder.glob("*_inference"))
    if len(inference_folders) == 0:
        logger.warning(f"No *_inference folders found in {model_folder}")
        return

    logger.info(f"Found {len(inference_folders)} inference folders: {[f.name for f in inference_folders]}")

    # Process each inference folder
    for inference_folder in inference_folders:
        pkl_files = list(inference_folder.glob("*.pkl"))
        if len(pkl_files) == 0:
            logger.warning(f"No .pkl files found in {inference_folder}")
            continue

        if len(pkl_files) > 1:
            logger.warning(f"Multiple pkl files in {inference_folder}, processing all")

        for pkl_file in pkl_files:
            npz_file = pkl_file.with_suffix(".npz")
            convert_pkl_to_npz(pkl_file, npz_file)

    logger.success("Conversion completed!")


if __name__ == "__main__":
    main()
