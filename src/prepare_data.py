from pathlib import Path

from .data_loader import load_data
from .preprocess import encode_data, handle_missing_values, handle_outliers, normalize_data
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_SEED = 42
DEFAULT_PREPROCESSING_VARIATIONS = [["missing", "outlier", "encode", "normalize"]]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "data_processed"

def split_data(dataframe, target_column=TARGET_COLUMN, test_size=TEST_SIZE, seed=RANDOM_SEED):
    x = dataframe.drop (columns = [target_column])
    y = dataframe[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    return x_train, x_test, y_train, y_test


def encode_target(target):
    encoded = (
        target.astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

    return encoded.astype("int64")


def _save_dataset(x, y, path):
    dataset = x.reset_index(drop=True).copy()
    dataset[TARGET_COLUMN] = y.reset_index(drop=True)
    dataset.to_csv(path, index=False)


def _apply_steps(x_train, x_test, steps, missing_type="Cut", outlier_type="Clip"):
    processors = {
        "encode": encode_data,
        "missing": handle_missing_values,
        "outlier": handle_outliers,
        "normalize": normalize_data,
    }

    for step in steps:
        if step not in processors:
            raise ValueError(f"Unknown preprocessing step: {step}")
        
        if step == "missing":
            result = processors[step](x_train, x_test, type=missing_type)
        elif step == "outlier":
            result = processors[step](x_train, x_test, type=outlier_type)
        else:
            result = processors[step](x_train, x_test)

        x_train, x_test = result

    return x_train, x_test


def prepare_data(preprocessing_variations=None, output_dir=DEFAULT_OUTPUT_DIR, missing_type="Cut", outlier_type="Clip"):
    if preprocessing_variations is None:
        preprocessing_variations = [steps.copy() for steps in DEFAULT_PREPROCESSING_VARIATIONS]
    elif preprocessing_variations and isinstance(preprocessing_variations[0], str):
        preprocessing_variations = [preprocessing_variations]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    x_train, x_test, y_train, y_test = split_data(data, target_column=TARGET_COLUMN)

    y_train = encode_target(y_train)
    y_test = encode_target(y_test)

    _save_dataset(x_test, y_test, output_dir / "test_raw.csv")

    prepared = {}
    for steps in preprocessing_variations:
        steps = list(steps)
        name = "_".join(steps) if steps else "raw"
        train_x, test_x = _apply_steps(x_train.copy(), x_test.copy(), steps, missing_type=missing_type, outlier_type=outlier_type)
        train_y = y_train.copy().loc[train_x.index]
        test_y = y_test.copy().loc[test_x.index]

        _save_dataset(train_x, train_y, output_dir / f"train_{name}.csv")
        _save_dataset(test_x, test_y, output_dir / f"test_{name}.csv")
        prepared[name] = (train_x, test_x, train_y, test_y)

    return prepared


if __name__ == "__main__":
    prepare_data()
