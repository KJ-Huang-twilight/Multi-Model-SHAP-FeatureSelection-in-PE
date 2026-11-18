"""
Deep Neural Network (DNN) model training with Optuna hyperparameter tuning,
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
from sklearn.preprocessing import StandardScaler
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras import metrics
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
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
np.random.seed(seed)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Ensure numpy arrays are float32 for Keras
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)


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
# Build DNN model
# ---------------------------------------------------------------------------
def create_model(neurons, optimizer, activation, dropout_rate, input_dim, l2_reg=0.01):
    inputs = Input(shape=(input_dim,))
    a = inputs
    for i, neuron in enumerate(neurons):
        a = Dense(neuron, activation=activation, kernel_regularizer=l2(l2_reg))(a)
        # Add BatchNormalization in all but the last hidden layer to stabilize training
        if i < len(neurons) - 1:
            a = BatchNormalization()(a)
        a = Dropout(dropout_rate)(a)
    outputs = Dense(1, activation="sigmoid")(a)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.Precision(), metrics.Recall()],
    )
    return model


# ---------------------------------------------------------------------------
# Custom callback for per-epoch AUC tracking
# ---------------------------------------------------------------------------
class AUCCallback(Callback):
    def __init__(self, training_data, validation_data):
        self.training_data = training_data
        self.validation_data = validation_data
        self.train_auc_values = []
        self.val_auc_values = []

    def on_epoch_end(self, epoch, logs=None):
        x_tr, y_tr = self.training_data
        y_tr_prob = self.model.predict(x_tr, verbose=0).flatten()
        self.train_auc_values.append(roc_auc_score(y_tr, y_tr_prob))

        x_val, y_val = self.validation_data
        y_val_prob = self.model.predict(x_val, verbose=0).flatten()
        self.val_auc_values.append(roc_auc_score(y_val, y_val_prob))

        print(
            f"Epoch {epoch + 1}: "
            f"Train AUC = {self.train_auc_values[-1]:.4f}, "
            f"Val AUC = {self.val_auc_values[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Optuna objective for DNN hyperparameter search
# ---------------------------------------------------------------------------
def objective(trial):
    layers = trial.suggest_int("layers", 1, 5)
    neurons = [trial.suggest_int(f"neurons_{i}", 16, 128) for i in range(layers)]
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    epochs = trial.suggest_int("epochs", 20, 80)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 0.001, 0.1)

    # Split training data into train and validation subsets for tuning
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train, y_train, test_size=0.3, random_state=seed
    )

    model = create_model(
        neurons, optimizer, activation, dropout_rate, x_train.shape[1], l2_reg
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
    )

    model.fit(
        x_train_split,
        y_train_split,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_split, y_val_split),
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    y_val_pred_prob = model.predict(x_val_split, verbose=0).flatten()
    return roc_auc_score(y_val_split, y_val_pred_prob)


# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, n_jobs=1)

best_params = study.best_params
print("Best hyperparameters:", best_params)

# Build final model with best hyperparameters
best_model = create_model(
    neurons=[best_params[f"neurons_{i}"] for i in range(best_params["layers"])],
    optimizer=best_params["optimizer"],
    activation=best_params["activation"],
    dropout_rate=best_params["dropout_rate"],
    input_dim=x_train.shape[1],
    l2_reg=best_params["l2_reg"],
)

# Callbacks for final training
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=10, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
auc_callback = AUCCallback(
    training_data=(x_train, y_train), validation_data=(x_test, y_test)
)

# Train final model
history = best_model.fit(
    x_train,
    y_train,
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    validation_split=0.3,
    callbacks=[early_stopping, reduce_lr, auc_callback],
    verbose=1,
)

# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve-Loss")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_learning_curve_loss.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Learning Curve-Accuracy")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_learning_curve_accuracy.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(auc_callback.train_auc_values, label="Training AUC")
plt.plot(auc_callback.val_auc_values, label="Validation AUC")
plt.xlabel("Epochs")
plt.ylabel("AUC")
plt.title("Learning Curve-AUC")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_learning_curve_auc.png"), dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# Evaluation on training set with 95% CIs
# ---------------------------------------------------------------------------
y_pred_proba = best_model.predict(x_train, verbose=0).flatten()
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
y_pred_proba_test = best_model.predict(x_test, verbose=0).flatten()
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
# SHAP analysis (DeepExplainer)
# ---------------------------------------------------------------------------
background_data = x_train[:5000]  # Background sample for SHAP

explainer = shap.DeepExplainer(best_model, background_data)
shap_values = explainer.shap_values(x_train)

print(f"SHAP values shape: {np.array(shap_values).shape}")

feature_labels_list = list(feature_labels)

# Squeeze last dimension: (n_samples, n_features)
shap_values_2d = np.squeeze(shap_values)

# SHAP summary plot
shap.summary_plot(shap_values_2d, x_train, feature_names=feature_labels_list)

# Compute and save SHAP feature importance
shap_importances = np.mean(np.abs(shap_values_2d), axis=0)
shap_sorted_indices = np.argsort(shap_importances)[::-1]
shap_sorted_feature_labels = np.array(feature_labels_list)[shap_sorted_indices]
shap_sorted_scores = shap_importances[shap_sorted_indices]

shap_feature_importance = pd.DataFrame(
    {"Feature": shap_sorted_feature_labels, "Importance": shap_sorted_scores}
)
shap_feature_importance.to_excel(
    os.path.join(OUTPUT_DIR, "14-dnn-shap.xlsx"), index=False
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
plt.savefig(os.path.join(OUTPUT_DIR, "14_DNN_shap_bar.png"), dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# Export results to a Word document
# ---------------------------------------------------------------------------
doc_file_path = os.path.join(OUTPUT_DIR, "14_DNN.docx")

doc = Document()
doc.add_heading("14_DNN", 0)

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