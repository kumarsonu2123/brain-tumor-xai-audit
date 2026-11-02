from dataclasses import dataclass

import h5py
import numpy as np

LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


@dataclass
class TumorSample:
    label: int                # raw 1/2/3
    patient_id: str
    image: np.ndarray         # (H, W) int16
    tumor_mask: np.ndarray    # (H, W) uint8 in {0, 1}
    tumor_border: np.ndarray  # (K, 2) polygon vertices, assumed (row, col)


def _decode_chars(ds) -> str:
    # MATLAB char arrays come through h5py as uint16 code points
    return "".join(chr(int(c)) for c in np.asarray(ds).ravel())


def read_mat(path) -> TumorSample:
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        # h5py exposes MATLAB's column-major arrays with axes reversed
        image = np.asarray(cj["image"]).T
        mask = np.asarray(cj["tumorMask"]).T.astype(np.uint8)
        return TumorSample(
            label=int(np.asarray(cj["label"]).item()),
            patient_id=_decode_chars(cj["PID"]).strip(),
            image=image,
            tumor_mask=mask,
            tumor_border=np.asarray(cj["tumorBorder"]).ravel().reshape(-1, 2),
        )