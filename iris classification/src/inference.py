import joblib
import numpy as np
from pathlib import Path


def _default_model_path():
    # Resolve path relative to the project root (parent of `src`)
    return Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"


def predict(sample, model_path: str | Path | None = None):
    """Predict class for a sample or batch.

    Args:
        sample: list or array-like of shape (4,) or (n,4)
        model_path: optional path to model file; if None uses project `models/best_model.pkl`
    """
    model_file = Path(model_path) if model_path is not None else _default_model_path()
    model = joblib.load(str(model_file))
    arr = np.array(sample)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return model.predict(arr)


if __name__ == '__main__':
    example = [5.1, 3.5, 1.4, 0.2]
    print(predict(example))
