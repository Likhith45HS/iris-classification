"""Train classifiers on the Iris dataset and save the best model.

Saves:
 - models/best_model.pkl
 - models/metrics.txt
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models (use pipelines where scaling helps)
models = {
    'knn': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=3))]),
    'logreg': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=200))]),
    'dt': DecisionTreeClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    results[name] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'confusion_matrix': cm.tolist(),
        'report': report
    }
    print(f"Model: {name}")
    print(report)

# Choose best model by accuracy
best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_model = models[best_name]

# Save best model
model_path = MODELS_DIR / 'best_model.pkl'
joblib.dump(best_model, model_path)

# Save metrics summary
metrics_path = MODELS_DIR / 'metrics.txt'
with metrics_path.open('w', encoding='utf-8') as f:
    f.write(f"best_model: {best_name}\n\n")
    for name, r in results.items():
        f.write(f"=== {name} ===\n")
        f.write(f"accuracy: {r['accuracy']:.4f}\n")
        f.write(f"precision: {r['precision']:.4f}\n")
        f.write(f"recall: {r['recall']:.4f}\n")
        f.write("confusion_matrix:\n")
        for row in r['confusion_matrix']:
            f.write(' '.join(map(str, row)) + "\n")
        f.write('\n')
    f.write('\n')

print(f"Saved best model '{best_name}' to: {model_path}")
print(f"Saved metrics to: {metrics_path}")
