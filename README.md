# Multi-Model-SHAP-FeatureSelection-in-PE

This repository contains Python scripts for the 20-model machine-learning and SHAP interpretability pipeline used in our study. The pipeline systematically identifies consensus core genes associated with Preeclampsia (PE).

## Code structure

### 01_Random Forest.py ~ 20_ANN.py
The 20 independent machine learning model training scripts. Each script performs data loading, Optuna hyperparameter optimization, model training, evaluation, and generates feature importance rankings. A consistent 80/20 train-test split and fixed random seed are applied across all models.

### Feature Importance Output Formats: 

- **Dual output (Model-intrinsic + SHAP).py**

  Scripts 01 through 12 (Random Forest, Gradient Boosting Trees, XGBoost, LightGBM, CatBoost, NGBoost, Extremely Randomized Trees, Oblique Random Forest, Logistic Regression, Lasso Regression, Ridge Regression, ElasticNetCV) output both their native feature importance rankings (e.g., feature_importances_, coefficients) and the SHAP-based importance rankings (SHAP-TreeExplainer or LinearExplainer).

- **SHAP-only output.py**

  Scripts 13 through 20 (MLP, DNN, SVM, Bayes, GPC, KNN, TPOT, ANN) output only the SHAP-based importance rankings (utilizing SHAP-DeepExplainer or KernelExplainer), as they do not provide native, directly comparable feature importance metrics.

### Rank_aggregate.py
The rank aggregation script. It computes the consensus ranking for features by calculating the average rank of each gene across multiple models. It can be run separately on the two sets of ranking files:

- Aggregating all **Model-intrinsic** ranking tables (from scripts 01–12).

- Aggregating all **SHAP-based** ranking tables (from all 20 scripts).

### Rank_intersect.py

The final core gene discovery script. It takes the top 100 genes from the **"Model-intrinsic consensus ranking"** and the **"SHAP-based consensus ranking"**, identifies their intersection, and re-ranks the overlapping genes by their combined average rank to determine the final disease-associated core gene set.

## Notes
- This repository contains only the Python implementation of the machine learning and feature ranking pipeline. Downstream analyses (such as single-cell RNA-seq, deconvolution, and clinical model validation) are performed using R as detailed in the manuscript.
- Before running, you must change the DATA_FILE_PATH variable at the top of each script to match your local dataset location. The scripts expect the label column to be named "lable".
- The pipeline requires standard data science and machine learning libraries, including scikit-learn, optuna, shap, pandas, numpy, and model-specific packages (xgboost, lightgbm, catboost, ngboost, treeple, tpot, tensorflow).
