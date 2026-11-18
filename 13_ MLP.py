"""
MLP (Multi-Layer Perceptron) model training with Optuna hyperparameter
tuning, SHAP interpretation, and 95% confidence intervals for performance
metrics.

"""

import os
import warnings

import numpy as np
import pandas as pd
import optuna
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
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
tf.random.set_seed(seed)
np.random.seed(seed)

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
# Custom cross-validation function for Keras models
# ---------------------------------------------------------------------------
def cross_validate_model(x_train, y_train, model, cv_folds=5, metric=roc_auc_score):
    scores = []
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Predict
    y_pred = model.predict(x_train).flatten()

    # Compute the specified evaluation metric
    score = metric(y_train, y_pred)
    scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Build an MLP model from the given hyperparameters
# ---------------------------------------------------------------------------
def build_model(input_dim, learning_rate, hidden_units, layers):
    input_layer = Input(shape=(input_dim,))
    a = input_layer
    for _ in range(layers):
        a = Dense(hidden_units, activation="relu")(a)
    output_layer = Dense(1, activation="sigmoid")(a)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Optuna objective for MLP hyperparameter search
# ---------------------------------------------------------------------------
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_units = trial.suggest_int("hidden_units", 20, 400)
    layers = trial.suggest_int("layers", 1, 5)

    model = build_model(x_train.shape[1], learning_rate, hidden_units, layers)
    scores = cross_validate_model(x_train, y_train, model, cv_folds=5)
    return np.mean(scores)


# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

best_params = study.best_params
print("Best hyperparameters:", best_params)

# Build and train final model with best hyperparameters
best_model = build_model(
    x_train.shape[1],
    best_params["learning_rate"],
    best_params["hidden_units"],
    best_params["layers"],
)
best_model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

# ---------------------------------------------------------------------------
# Evaluation on training set with 95% CIs
# ---------------------------------------------------------------------------
y_pred_proba = best_model.predict(x_train).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

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
y_pred_proba_test = best_model.predict(x_test).flatten()
y_pred_test = (y_pred_proba_test > 0.5).astype(int)

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
# MLP weight-based feature importance
# ---------------------------------------------------------------------------
feature_importances = np.mean(np.abs(best_model.layers[-1].get_weights()[0]), axis=1)

sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_labels = np.array(feature_labels)[sorted_indices]
sorted_scores = feature_importances[sorted_indices]

feature_importance_df = pd.DataFrame(
    {"Feature": sorted_feature_labels, "Importance": sorted_scores}
)
feature_importance_df.to_excel(os.path.join(OUTPUT_DIR, "13-mlp-00.xlsx"), index=False)

# ---------------------------------------------------------------------------
# SHAP analysis (DeepExplainer)
# ---------------------------------------------------------------------------
feature_labels = list(feature_labels)

# Summarize background data for SHAP
background_data = shap.sample(x_train, 5000).values

explainer = shap.DeepExplainer(best_model, background_data)
shap_values = explainer.shap_values(background_data)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("Shape of shap_values[0]:", np.array(shap_values[0]).shape)
print("Shape of background_data_summary:", background_data.shape)

assert shap_values.shape[1] == background_data.shape[1], (
    "Feature names length does not match SHAP values dimensions."
)

# Squeeze last dimension: (n_samples, n_features)
shap_values_2d = np.squeeze(shap_values)

# SHAP summary plot
shap.summary_plot(shap_values_2d, background_data, feature_names=feature_labels)

# Compute and save SHAP feature importance
shap_importances = np.mean(np.abs(shap_values_2d), axis=0)
shap_sorted_indices = np.argsort(shap_importances)[::-1]
shap_sorted_feature_labels = np.array(feature_labels)[shap_sorted_indices]
shap_sorted_scores = shap_importances[shap_sorted_indices]

shap_feature_importance = pd.DataFrame(
    {"Feature": shap_sorted_feature_labels, "Importance": shap_sorted_scores}
)
shap_feature_importance.to_excel(
    os.path.join(OUTPUT_DIR, "13-mlp-shap.xlsx"), index=False
)

# Custom colormap for SHAP bar plot
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B8D0E7", "#3574B1"])

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values_2d, x_train, feature_names=feature_labels,
    plot_type="bar", show=False,
)

bars = ax.patches
for i, bar in enumerate(bars):
    bar.set_color(cmap(i / len(bars)))

ax.set_ylabel(ax.get_ylabel(), fontsize=14, weight="bold")
ax.tick_params(axis="both", which="major", labelsize=12)
ax.spines["left"].set_linewidth(2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_mlp_shap_bar.png"), dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# Export results to a Word document
# ---------------------------------------------------------------------------
doc_file_path = os.path.join(OUTPUT_DIR, "13_MLP.docx")

doc = Document()
doc.add_heading("13_MLP", 0)

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