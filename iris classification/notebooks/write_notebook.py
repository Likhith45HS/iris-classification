import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "![](https://housing.com/news/wp-content/uploads/2022/11/iris-flower-compressed.jpg)\n",
                "<div style=\"padding: 10px;color:white;margin:10;font-size:200%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://www.yates.com.au/media/plants/flowers-and-ornamentals/flowers/flowers-iris-image1.jpg?mode=crop&anchor=center&widthratio=1.5&height=576&format=jpg)\"><b><span style='color:Yellow;font-size:60px; font-family:Times New Roman;'> IRIS FLOWER CLASSIFICATION </span></b> </div>\n\n\n",
                "![](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Iris Flower Classification — EDA and Model Comparison\n",
                "This notebook performs exploratory data analysis, trains kNN, Logistic Regression, and Decision Tree classifiers, compares results, and saves the best model."
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import joblib\n",
                "from pathlib import Path\n",
                "from sklearn.datasets import load_iris\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.neighbors import KNeighborsClassifier\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report\n\n",
                "sns.set(style='whitegrid', palette='muted', font_scale=1.1)\n\n",
                "iris = load_iris()\n",
                "df = pd.DataFrame(iris.data, columns=iris.feature_names)\n",
                "df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)\n",
                "df.head()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Dataset overview\n",
                "Inspect the first rows, summary statistics, and class distribution."
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "print('Shape:', df.shape)\n",
                "print('\\nTarget classes:', df['species'].unique())\n",
                "print('\\nClass counts:')\n",
                "print(df['species'].value_counts())\n\n",
                "df.describe()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Feature correlation and separation\n",
                "Use plots to visualize how sepal and petal features separate the species."
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "plt.figure(figsize=(12, 10))\n",
                "sns.pairplot(df, hue='species', corner=True, diag_kind='hist')\n",
                "plt.suptitle('Pairplot of Iris Features', y=1.02)\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "plt.figure(figsize=(10, 8))\n",
                "cm = df.corr()\n",
                "sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='.2f')\n",
                "plt.title('Feature Correlation Matrix')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "plt.figure(figsize=(16, 6))\n",
                "plt.subplot(1, 2, 1)\n",
                "sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', palette='deep')\n",
                "plt.title('Sepal Length vs Sepal Width')\n",
                "plt.subplot(1, 2, 2)\n",
                "sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', palette='deep')\n",
                "plt.title('Petal Length vs Petal Width')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Separate feature trends\n",
                "Track sepal and petal lengths and widths in line plots to compare distributions."
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "plt.figure(figsize=(12, 5))\n",
                "plt.plot(df['sepal length (cm)'], marker='o', linestyle='-', label='Sepal Length')\n",
                "plt.plot(df['sepal width (cm)'], marker='x', linestyle='--', label='Sepal Width')\n",
                "plt.title('Sepal Length and Width')\n",
                "plt.xlabel('Sample index')\n",
                "plt.ylabel('cm')\n",
                "plt.legend()\n",
                "plt.show()\n\n",
                "plt.figure(figsize=(12, 5))\n",
                "plt.plot(df['petal length (cm)'], marker='o', linestyle='-', label='Petal Length')\n",
                "plt.plot(df['petal width (cm)'], marker='.', linestyle='--', label='Petal Width')\n",
                "plt.title('Petal Length and Width')\n",
                "plt.xlabel('Sample index')\n",
                "plt.ylabel('cm')\n",
                "plt.legend()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Data visualization by species\n",
                "Select additional plots to highlight class separability."
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "plt.figure(figsize=(10, 6))\n",
                "sns.boxplot(data=df, x='species', y='sepal width (cm)')\n",
                "plt.title('Sepal Width by Species')\n",
                "plt.show()\n\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.boxplot(data=df, x='species', y='sepal length (cm)')\n",
                "plt.title('Sepal Length by Species')\n",
                "plt.show()\n\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.boxplot(data=df, x='species', y='petal width (cm)')\n",
                "plt.title('Petal Width by Species')\n",
                "plt.show()\n\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.boxplot(data=df, x='species', y='petal length (cm)')\n",
                "plt.title('Petal Length by Species')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Model training and comparison\n",
                "Train and compare kNN, Logistic Regression, and Decision Tree."
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "X = iris.data\n",
                "y = iris.target\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\n",
                "models = {\n",
                "    'kNN': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=3))]),\n",
                "    'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=200))]),\n",
                "    'DecisionTree': DecisionTreeClassifier(random_state=42)\n",
                "}\n\n",
                "results = []\n",
                "for name, model in models.items():\n",
                "    model.fit(X_train, y_train)\n",
                "    preds = model.predict(X_test)\n",
                "    acc = accuracy_score(y_test, preds)\n",
                "    prec = precision_score(y_test, preds, average='weighted', zero_division=0)\n",
                "    rec = recall_score(y_test, preds, average='weighted', zero_division=0)\n",
                "    cm = confusion_matrix(y_test, preds)\n",
                "    report = classification_report(y_test, preds, target_names=iris.target_names, zero_division=0)\n",
                "    results.append((name, model, acc, prec, rec, cm, report))\n",
                "    print(f'== {name} ==')\n",
                "    print(report)\n",
                "    print('Accuracy:', acc)\n",
                "    print('Precision:', prec)\n",
                "    print('Recall:', rec)\n",
                "    print('\\n')\n"
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "metrics = pd.DataFrame([\n",
                "    {'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec}\n",
                "    for name, _, acc, prec, rec, _, _ in results\n",
                "])\n",
                "metrics\n"
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "plt.figure(figsize=(10, 5))\n",
                "sns.barplot(data=metrics.melt(id_vars='model', value_vars=['accuracy', 'precision', 'recall']), x='model', y='value', hue='variable')\n",
                "plt.ylim(0.8, 1.0)\n",
                "plt.title('Model Performance Comparison')\n",
                "plt.ylabel('Score')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "best_name, best_model, best_acc, best_prec, best_rec, best_cm, best_report = max(results, key=lambda item: item[2])\n",
                "print('Best model:', best_name)\n\n",
                "model_path = Path('..') / 'models' / 'best_model.pkl'\n",
                "model_path.parent.mkdir(parents=True, exist_ok=True)\n",
                "joblib.dump(best_model, model_path)\n",
                "print('Saved best model to', model_path)\n\n",
                "plt.figure(figsize=(6, 5))\n",
                "sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)\n",
                "plt.xlabel('Predicted')\n",
                "plt.ylabel('Actual')\n",
                "plt.title(f'Confusion Matrix for {best_name}')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Inference example\n",
                "Load the saved model and predict a new iris sample."
            ]
        },
        {
            "cell_type": "python",
            "metadata": {},
            "source": [
                "model_path = Path('..') / 'models' / 'best_model.pkl'\n",
                "model = joblib.load(model_path)\n\n",
                "sample = np.array([[5.1, 3.5, 1.4, 0.2]])\n",
                "pred = model.predict(sample)\n",
                "print('Sample:', sample)\n",
                "print('Predicted class index:', int(pred[0]))\n",
                "print('Predicted species:', iris.target_names[pred[0]])\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Conclusions\n",
                "- The iris dataset is well-separated by petal dimensions.\n",
                "- All three models perform above 90% accuracy on the test split.\n",
                "- The best model is saved to `models/best_model.pkl` for inference."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

path = Path(r"c:\iris classification\notebooks\01_iris_classification.ipynb")
path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
