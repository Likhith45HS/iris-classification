import joblib
import numpy as np
import tempfile
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def _default_model_path():
    # Resolve path relative to the project root (parent of `src`)
    return Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"


def _temp_model_path():
    temp_dir = Path(tempfile.gettempdir()) / "iris_classification"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / "best_model.pkl"


def _train_default_model(model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=200))
    ])
    model.fit(X, y)
    joblib.dump(model, str(model_path))
    return model


def predict(sample, model_path: str | Path | None = None):
    """Predict class for a sample or batch.

    Args:
        sample: list or array-like of shape (4,) or (n,4)
        model_path: optional path to model file; if None uses project `models/best_model.pkl`
    """
    model_file = Path(model_path) if model_path is not None else _default_model_path()

    if not model_file.exists():
        try:
            model = _train_default_model(model_file)
        except OSError:
            temp_path = _temp_model_path()
            if not temp_path.exists():
                model = _train_default_model(temp_path)
            else:
                model = joblib.load(str(temp_path))
    else:
        model = joblib.load(str(model_file))

    arr = np.array(sample)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return model.predict(arr)


if __name__ == '__main__':
    example = [5.1, 3.5, 1.4, 0.2]
    print(predict(example))
