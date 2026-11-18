"""
ElasticNetCV model training with Optuna hyperparameter tuning, SHAP
interpretation, and 95% confidence intervals for performance metrics.

"""

import os
import warnings

import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, train_test_split
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

x = data[feature_labels]
y = data["lable"]

seed = 42
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)


# ---------------------------------------------------------------------------
# Performance metrics with 95% confidence intervals (bootstrap)
# ---------------------------------------------------------------------------
def calculate_metrics_with_ci(y_true, y_pred, y_score, n_resamples=2000):
    """Compute point estimates and 95% bootstrap CIs for classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted hard labels.
    y_score : array-like
        Continuous scores for the positive class (used for ROC AUC).
    n_resamples : int
        Number of bootstrap resamples.

    Returns
    -------
    dict
        Point estimates and (low, high) confidence intervals for each metric.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    metrics_point = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
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

    auc_ci = safe_bootstrap(auc_fn, y_true, y_score)
    out["roc_auc"] = metrics_point["roc_auc"]
    out["roc_auc_ci"] = (auc_ci.low, auc_ci.high)

    return out


# ---------------------------------------------------------------------------
# Optuna objective for ElasticNet hyperparameter search
# ---------------------------------------------------------------------------
def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.2, 1.0)
    en = ElasticNet(max_iter=10000, random_state=seed, alpha=alpha, l1_ratio=l1_ratio)
    cv_scores = cross_val_score(
        en, x_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    return np.mean(cv_scores)


# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train final model with best hyperparameters
best_model = ElasticNet(
    alpha=best_params["alpha"],
    l1_ratio=best_params["l1_ratio"],
    max_iter=10000,
    random_state=seed,
)
best_model.fit(x_train, y_train)

# ---------------------------------------------------------------------------
# Evaluation on training set with 95% CIs
# ---------------------------------------------------------------------------
y_pred_continuous = best_model.predict(x_train)        # continuous scores for AUC
y_pred = (y_pred_continuous > 0.5).astype(int)         # hard labels for other metrics

train_metrics = calculate_metrics_with_ci(y_train, y_pred, y_pred_continuous)

fpr, tpr, thresholds = roc_curve(y_train, y_pred_continuous)
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
y_pred_continuous_test = best_model.predict(x_test)
y_pred_test = (y_pred_continuous_test > 0.5).astype(int)

test_metrics = calculate_metrics_with_ci(y_test, y_pred_test, y_pred_continuous_test)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_continuous_test)
cm_test = confusion_matrix(y_test, y_pred_test)

print("=== Test set ===")
for name in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
    lo, hi = test_metrics[f"{name}_ci"]
    print(f"{name}: {test_metrics[name]:.4f} (95% CI: {lo:.4f}-{hi:.4f})")
print("Confusion Matrix:")
print(cm_test)

# ---------------------------------------------------------------------------
# ElasticNet coefficient-based feature importance
# ---------------------------------------------------------------------------
feature_importances = best_model.coef_

sorted_indices = np.argsort(np.abs(feature_importances))[::-1]
sorted_feature_labels = np.array(feature_labels)[sorted_indices]
sorted_scores = feature_importances[sorted_indices]

feature1 = pd.DataFrame({"Feature": sorted_feature_labels, "Importance": sorted_scores})
feature1.to_excel(os.path.join(OUTPUT_DIR, "12-en-00.xlsx"), index=False)

# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------
explainer = shap.LinearExplainer(best_model, x_train)
shap_values = explainer.shap_values(x_train)

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
    os.path.join(OUTPUT_DIR, "12-en-shap.xlsx"), index=False
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
plt.savefig(os.path.join(OUTPUT_DIR, "12_ElasticNet_shap_bar.png"), dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# Export results to a Word document
# ---------------------------------------------------------------------------
doc_file_path = os.path.join(OUTPUT_DIR, "12_ElasticNetCV.docx")

doc = Document()
doc.add_heading("12_ElasticNetCV", 0)

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