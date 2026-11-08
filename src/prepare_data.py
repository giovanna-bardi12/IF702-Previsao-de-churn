from pathlib import Path

from .data_loader import RANDOM_SEED, TARGET_COLUMN, load_data, split_data
from .preprocess import encode_data, handle_missing_values, normalize_data


DEFAULT_PREPROCESSING_VARIATIONS = [["missing", "encode", "normalize"]]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "data_processed"

PROCESSORS = {
    "missing": handle_missing_values,
    "encode": encode_data,
    "normalize": normalize_data,
}



def encode_target(target):
    encoded = (
        target.astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )
    if encoded.isna().any():
        raise ValueError("Valores inesperados no alvo (esperado 'Yes'/'No')")

    return encoded.astype("int64")


def _save_dataset(x, y, path):
    dataset = x.reset_index(drop=True).copy()
    dataset[TARGET_COLUMN] = y.reset_index(drop=True)
    dataset.to_csv(path, index=False)


def _apply_steps(x_train, x_val,x_test, steps, missing_type="Zero"):
    for step in steps:
        if step not in PROCESSORS:
            raise ValueError(f"Unknown preprocessing step: {step}")
        
        if step == "missing":
            result = PROCESSORS[step](x_train, x_val, x_test, type=missing_type)
        else:
            result = PROCESSORS[step](x_train, x_val, x_test)

        x_train,x_val ,x_test = result

    return x_train, x_val, x_test


def prepare_data(preprocessing_variations=None, output_dir=DEFAULT_OUTPUT_DIR,data_path=None, missing_type="Zero",seed=RANDOM_SEED):
    if preprocessing_variations is None:
        preprocessing_variations = [list(s) for s in DEFAULT_PREPROCESSING_VARIATIONS]
    elif preprocessing_variations and isinstance(preprocessing_variations[0], str):
        preprocessing_variations = [preprocessing_variations]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data, target_column=TARGET_COLUMN, seed=seed)

    y_train = encode_target(y_train)
    y_val = encode_target(y_val)
    y_test = encode_target(y_test)


    # Conjuntos brutos (só divididos), úteis para conferência
    for split, x, y in (("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)):
        _save_dataset(x, y, output_dir / f"{split}_raw.csv")


    prepared = {}
    for steps in preprocessing_variations:
        steps = list(steps)
        name = "_".join(steps) if steps else "raw"

        px_train, px_val, px_test = _apply_steps(
            x_train, x_val, x_test, steps, missing_type=missing_type
        )

        for split, x, y in (("train", px_train, y_train), ("val", px_val, y_val), ("test", px_test, y_test)):
            _save_dataset(x, y, output_dir / f"{split}_{name}.csv")
 
        prepared[name] = (px_train, px_val, px_test, y_train, y_val, y_test)
 
    return prepared

if __name__ == "__main__":
    prepare_data()
