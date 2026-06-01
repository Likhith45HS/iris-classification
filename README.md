# Iris Classification

This project performs Iris flower classification using classic features: sepal/petal length and width.

It includes:
- EDA and feature visualizations
- Training and comparison of kNN, Logistic Regression, and Decision Tree
- Accuracy, precision, recall, and confusion matrix reporting
- Best model save as `models/best_model.pkl`
- Example inference code

## Setup

From `c:\iris classification`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the notebook

Open and run `notebooks/01_iris_classification.ipynb`.
The notebook loads the dataset, performs EDA, trains the models, and saves the best model.

## Run the script

To train and save the best model from code directly:

```powershell
python src\train_and_save.py
```

This produces:
- `models/best_model.pkl`
- `models/metrics.txt`

## Inference example

```python
import joblib
import numpy as np
model = joblib.load('models/best_model.pkl')
example = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = model.predict(example)
print('Predicted species index:', pred[0])
```

Or use the helper:

```python
from src.inference import predict
print(predict([5.1, 3.5, 1.4, 0.2]))
```

## GitHub push

If you want to push to GitHub, run:

```powershell
git init
git add .
git commit -m "Add iris classification project"
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```

Replace `<your-github-repo-url>` with the repository URL.
