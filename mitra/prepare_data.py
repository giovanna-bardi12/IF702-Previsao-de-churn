"""Prepare MITRA-specific splits without changing the other models' data."""
from mitra.common import MITRA_DATA_DIR
from src.prepare_data import prepare_data


if __name__ == "__main__":
    prepare_data(output_dir=MITRA_DATA_DIR, oversample_train=False, oversample_val=False)
    print(f"MITRA splits saved to {MITRA_DATA_DIR}")
