from __future__ import annotations

from pathlib import Path

import numpy as np


def save_latent_npz(data: dict[str, object], npz_path: str | Path) -> None:
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    npz_data: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if isinstance(value, list):
            np_list = []
            for item in value:
                if hasattr(item, "cpu"):
                    np_list.append(item.cpu().numpy())
                else:
                    np_list.append(np.asarray(item))
            npz_data[key] = np.concatenate(np_list, axis=0)
        else:
            if hasattr(value, "cpu"):
                npz_data[key] = value.cpu().numpy()
            else:
                npz_data[key] = np.asarray(value)

    np.savez(npz_path, **npz_data)

    txt_path = npz_path.with_suffix(".txt")
    with txt_path.open("w") as f:
        f.write(f"NPZ file: {npz_path.name}\n")
        f.write(f"Total keys: {len(npz_data)}\n\n")
        f.write("Keys and Shapes:\n")
        for key, value in npz_data.items():
            f.write(f"  {key}: {value.shape} (dtype: {value.dtype})\n")
