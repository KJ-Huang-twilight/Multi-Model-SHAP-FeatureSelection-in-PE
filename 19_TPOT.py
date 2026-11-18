"""
TPOT (Tree-based Pipeline Optimization Tool) auto-ML model training,
SHAP interpretation, and 95% confidence intervals for performance metrics.


Because TPOT may select any estimator, the script inspects the fitted pipeline
to decide whether to use `predict_proba` or `decision_function` for scores
needed by ROC AUC and SHAP.
"""

import os
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
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
from tpot import TPOTClassifier
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
# Utility: obtain continuous scores from any fitted pipeline
# ---------------------------------------------------------------------------
def get_scores(model, x_data):
    """Return continuous scores for the positive class.

    Tries `predict_proba` first, then `decision_function`; raises if neither
    is available. Used for ROC AUC computation and bootstrap resampling.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_data)
        return np.asarray(scores).ravel()
    raise AttributeError(
        "Fitted TPOT pipeline exposes neither predict_proba nor decision_function"
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
# Run TPOT auto-ML
# ---------------------------------------------------------------------------
tpot = TPOTClassifier(
    generations=10,
    population_size=50,
    verbosity=2,
    scoring="roc_auc",
    random_state=seed,
    cv=5,
    n_jobs=-1,
    config_dict="TPOT light",
)
tpot.fit(x_train, y_train)

# Retrieve best pipeline
best_model = tpot.fitted_pipeline_

print("Best pipeline:")
print(best_model)

print("\nPipeline steps:")
for step_name, step_obj in best_model.named_steps.items():
    print(f"Step name: {step_name}, type: {type(step_obj).__name__}")

# ---------------------------------------------------------------------------
# Evaluation on training set with 95% CIs
# ---------------------------------------------------------------------------
y_pred = best_model.predict(x_train)
y_score = get_scores(best_model, x_train)

train_metrics = calculate_metrics_with_ci(y_train, y_pred, y_score)

fpr, tpr, thresholds = roc_curve(y_train, y_score)
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
y_pred_test = best_model.predict(x_test)
y_score_test = get_scores(best_model, x_test)

test_metrics = calculate_metrics_with_ci(y_test, y_pred_test, y_score_test)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_score_test)
cm_test = confusion_matrix(y_test, y_pred_test)

print("=== Test set ===")
for name in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
    lo, hi = test_metrics[f"{name}_ci"]
    print(f"{name}: {test_metrics[name]:.4f} (95% CI: {lo:.4f}-{hi:.4f})")
print("Confusion Matrix:")
print(cm_test)

# ---------------------------------------------------------------------------
# Traditional feature importance (if available on the final estimator)
# ---------------------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    feature_importances = best_model.feature_importances_

    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_labels = np.array(feature_labels)[sorted_indices]
    sorted_scores = feature_importances[sorted_indices]

    feature1 = pd.DataFrame(
        {"Feature": sorted_feature_labels, "Importance": sorted_scores}
    )
    feature1.to_excel(os.path.join(OUTPUT_DIR, "19-TPOT-00.xlsx"), index=False)
    print("Traditional feature importance saved to 19-TPOT-00.xlsx")
else:
    print("The final model does not have 'feature_importances_' attribute.")

# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------
# Find the final estimator inside the pipeline
final_estimator = None
for step_name, step_obj in best_model.named_steps.items():
    if hasattr(step_obj, "predict"):
        final_estimator = step_obj
        print(f"Final estimator: {step_name} ({type(final_estimator).__name__})")
        break

if final_estimator is not None:
    try:
        if hasattr(final_estimator, "coef_"):
            print("Using SHAP LinearExplainer...")
            explainer = shap.LinearExplainer(final_estimator, x_train)
            shap_values = explainer.shap_values(x_train)
        elif hasattr(final_estimator, "feature_importances_"):
            print("Using SHAP TreeExplainer...")
            explainer = shap.TreeExplainer(final_estimator)
            shap_values = explainer.shap_values(x_train)
        else:
            print("Using SHAP KernelExplainer...")
            explainer = shap.KernelExplainer(
                best_model.predict_proba,
                shap.sample(x_train, 100),
            )
            shap_values = explainer.shap_values(x_train)

        # SHAP summary plot
        shap.summary_plot(shap_values, x_train, feature_names=feature_labels)

        # Compute and save SHAP feature importance
        if isinstance(shap_values, list):
            shap_importances = np.mean(np.abs(shap_values[0]), axis=0)
        else:
            shap_importances = np.mean(np.abs(shap_values), axis=0)

        sorted_indices = np.argsort(shap_importances)[::-1]
        sorted_feature_labels = np.array(feature_labels)[sorted_indices]
        shap_sorted_scores = shap_importances[sorted_indices]

        shap_feature_importance = pd.DataFrame(
            {"Feature": sorted_feature_labels, "Importance": shap_sorted_scores}
        )
        shap_feature_importance.to_excel(
            os.path.join(OUTPUT_DIR, "19-TPOT-shap.xlsx"), index=False
        )
        print("SHAP feature importance saved to 19-TPOT-shap.xlsx")

        # Custom colormap for SHAP bar plot
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B8D0E7", "#3574B1"])

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, x_train, feature_names=feature_labels,
            plot_type="bar", show=False,
        )

        bars = ax.patches
        for i, bar in enumerate(bars):
            bar.set_color(cmap(i / len(bars)))

        ax.set_ylabel(ax.get_ylabel(), fontsize=14, weight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.spines["left"].set_linewidth(2)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "19_TPOT_shap_bar.png"), dpi=300)
        plt.show()

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Skipping SHAP analysis...")
else:
    print("No final estimator found; skipping SHAP analysis...")

# ---------------------------------------------------------------------------
# Export results to a Word document
# ---------------------------------------------------------------------------
doc_file_path = os.path.join(OUTPUT_DIR, "19_TPOT.docx")

doc = Document()
doc.add_heading("19_TPOT", 0)

# Model information
doc.add_heading("Model information", level=1)
doc.add_paragraph(f"Best pipeline type: {type(best_model).__name__}")
for step_name, step_obj in best_model.named_steps.items():
    doc.add_paragraph(f"Step {step_name}: {type(step_obj).__name__}")

# Training set results
doc.add_heading("Training set evaluation", level=1)
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
doc.add_heading("Test set evaluation", level=1)
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

# Export the best pipeline code produced by TPOT
exported_code = tpot.export()
with open(os.path.join(OUTPUT_DIR, "19_TPOT_best_pipeline.py"), "w") as f:
    f.write(exported_code)
print("Best pipeline code saved to 19_TPOT_best_pipeline.py")