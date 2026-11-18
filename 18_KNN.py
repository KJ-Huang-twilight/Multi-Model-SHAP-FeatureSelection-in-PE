"""
K-Nearest Neighbors (KNN) model training with Optuna hyperparameter tuning,
SHAP interpretation, and 95% confidence intervals for performance metrics.

"""

import os
import warnings

import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from scipy.stats import bootstrap
from docx import Document

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration: change DATA_FILE_PATH to point at your own dataset.
# The dataset must contain the label column "lable" and feature columns.
# ---------------------------------------------------------------------------
DATA_FILE_PATH = "data/330PE.xlsx"   # <-- change this path only
OUTPUT_DIR = "results"

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
data = pd.read_excel(DATA_FILE_PATH)

# Read the original header row to recover feature labels (all columns except the last)
data2 = pd.read_excel(DATA_FILE_PATH, header=None)
feature_labels = data2.iloc[0, :-1].tolist()

# Preprocess data: fill missing feature values with column means
x = data[feature_labels].fillna(data[feature_labels].mean())
y = data["lable"]

seed = 42
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)


# ---------------------------------------------------------------------------
# Performance metrics with 95% confidence intervals (bootstrap)
# ---------------------------------------------------------------------------
def calculate_metrics_with_ci(y_true, y_pred, y_pred_proba, n_resamples=2000):
    """Compute point estimates and 95% bootstrap CIs for classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted hard labels.
    y_pred_proba : array-like
        Predicted probabilities for the positive class (used for ROC AUC).
    n_resamples : int
        Number of bootstrap resamples.

    Returns
    -------
    dict
        Point estimates and (low, high) confidence intervals for each metric.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred_proba = np.asarray(y_pred_proba)

    metrics_point = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }

    def safe_bootstrap(stat_func, *arrays, method="BCa"):
        """Bootstrap with BCa, falling back to percentile on failure."""
        try:
            res = bootstrap(
                arrays,
                statistic=stat_func,
                paired=True,
                vectorized=False,
                random_state=seed,
                method=method,
                n_resamples=n_resamples,
            )
            return res.confidence_interval
        except Exception:
            # Fall back to percentile method if BCa fails
            res = bootstrap(
                arrays,
                statistic=stat_func,
                paired=True,
                vectorized=False,
                random_state=seed,
                method="percentile",
                n_resamples=n_resamples,
            )
            return res.confidence_interval

    # Metric statistic functions
    fns = {
        "accuracy": lambda a, b: accuracy_score(a, b),
        "recall": lambda a, b: recall_score(a, b),
        "precision": lambda a, b: precision_score(a, b),
        "f1": lambda a, b: f1_score(a, b),
    }

    out = {}
    for name, fn in fns.items():
        ci = safe_bootstrap(fn, y_true, y_pred)
        out[name] = metrics_point[name]
        out[f"{name}_ci"] = (ci.low, ci.high)

    # ROC AUC needs special handling (single class -> 0.5)
    def auc_fn(a, b):
        if len(np.unique(a)) < 2:
            return 0.5
        return roc_auc_score(a, b)

    auc_ci = safe_bootstrap(auc_fn, y_true, y_pred_proba)
    out["roc_auc"] = metrics_point["roc_auc"]
    out["roc_auc_ci"] = (auc_ci.low, auc_ci.high)

    return out


# ---------------------------------------------------------------------------
# Optuna objective for KNN hyperparameter search
# ---------------------------------------------------------------------------
def objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 5, 35)
    metric = trial.suggest_categorical("metric", ["minkowski", "manhattan"])
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 3)

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors, p=p, weights=weights, metric=metric
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for train_index, val_index in skf.split(x_train, y_train):
        x_train_fold = x_train.iloc[train_index]
        x_val_fold = x_train.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]
        model.fit(x_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(x_val_fold)[:, 1]
        scores.append(roc_auc_score(y_val_fold, y_pred_proba))

    return np.mean(scores)


# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train final model with best hyperparameters
knn = KNeighborsClassifier(
    n_neighbors=best_params["n_neighbors"],
    p=best_params["p"],
    weights=best_params["weights"],
    metric=best_params["metric"],
)
knn.fit(x_train, y_train)

# ---------------------------------------------------------------------------
# Evaluation on training set with 95% CIs
# ---------------------------------------------------------------------------
y_pred = knn.predict(x_train)
y_pred_proba = knn.predict_proba(x_train)[:, 1]

train_metrics = calculate_metrics_with_ci(y_train, y_pred, y_pred_proba)

fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
cm = confusion_matrix(y_train, y_pred)

print("=== Training set ===")
for name in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
    lo, hi = train_metrics[f"{name}_ci"]
    print(f"{name}: {train_metrics[name]:.4f} (95% CI: {lo:.4f}-{hi:.4f})")
print("Confusion Matrix:")
print(cm)

# ---------------------------------------------------------------------------
# Evaluation on test set with 95% CIs
# ---------------------------------------------------------------------------
y_pred_test = knn.predict(x_test)
y_pred_proba_test = knn.predict_proba(x_test)[:, 1]

test_metrics = calculate_metrics_with_ci(y_test, y_pred_test, y_pred_proba_test)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)
cm_test = confusion_matrix(y_test, y_pred_test)

print("=== Test set ===")
for name in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
    lo, hi = test_metrics[f"{name}_ci"]
    print(f"{name}: {test_metrics[name]:.4f} (95% CI: {lo:.4f}-{hi:.4f})")
print("Confusion Matrix:")
print(cm_test)

# ---------------------------------------------------------------------------
# SHAP analysis (KernelExplainer)
# ---------------------------------------------------------------------------
background_data = shap.sample(x_train, 100)
explainer = shap.KernelExplainer(knn.predict_proba, background_data)
shap_values = explainer.shap_values(x_train)

print(f"SHAP values shape: {np.array(shap_values).shape}")

# For binary classification, take SHAP values of the positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]
else:
    shap_values = shap_values[:, :, 1]

# SHAP summary plot
shap.summary_plot(shap_values, x_train, feature_names=feature_labels)

# Compute and save SHAP feature importance
shap_importances = np.mean(np.abs(shap_values), axis=0)
shap_sorted_indices = np.argsort(shap_importances)[::-1]
shap_sorted_feature_labels = np.array(feature_labels)[shap_sorted_indices]
shap_sorted_scores = shap_importances[shap_sorted_indices]

shap_feature_importance = pd.DataFrame(
    {"Feature": shap_sorted_feature_labels, "Importance": shap_sorted_scores}
)
shap_feature_importance.to_excel(
    os.path.join(OUTPUT_DIR, "18-knn-shap.xlsx"), index=False
)

# Custom colormap for SHAP bar plot
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B8D0E7", "#3574B1"])

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values, x_train, feature_names=feature_labels, plot_type="bar", show=False
)

bars = ax.patches
for i, bar in enumerate(bars):
    bar.set_color(cmap(i / len(bars)))

ax.set_ylabel(ax.get_ylabel(), fontsize=14, weight="bold")
ax.tick_params(axis="both", which="major", labelsize=12)
ax.spines["left"].set_linewidth(2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "18_KNN_shap_bar.png"), dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# Export results to a Word document
# ---------------------------------------------------------------------------
doc_file_path = os.path.join(OUTPUT_DIR, "18_KNN.docx")

doc = Document()
doc.add_heading("18_KNN", 0)

doc.add_paragraph(f"Best hyperparameters: {best_params}")

# Training set results
doc.add_paragraph("--- Training set ---")
for name, label in [
    ("roc_auc", "ROC AUC"),
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]:
    lo, hi = train_metrics[f"{name}_ci"]
    doc.add_paragraph(
        f"{label}: {train_metrics[name]:.4f} (95% CI: {lo:.4f}-{hi:.4f})"
    )

doc.add_paragraph("Confusion Matrix:")
cm_str = str(cm).replace("\n", "\n    ")
doc.add_paragraph(f"    {cm_str}")

# Test set results
doc.add_paragraph("--- Test set ---")
for name, label in [
    ("roc_auc", "ROC AUC"),
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]:
    lo, hi = test_metrics[f"{name}_ci"]
    doc.add_paragraph(
        f"{label}: {test_metrics[name]:.4f} (95% CI: {lo:.4f}-{hi:.4f})"
    )

doc.add_paragraph("Confusion Matrix:")
cm_str_test = str(cm_test).replace("\n", "\n    ")
doc.add_paragraph(f"    {cm_str_test}")

doc.save(doc_file_path)
print(f"Evaluation metrics saved to {doc_file_path}")