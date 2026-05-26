#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ernest Namdar
"""

# Install needed packages if not yet installed
# pip install radMLBench scikit-learn lightgbm shap scipy

import os
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
)
import numpy as np
import torch
import random
import logging
import shap
from itertools import product
import matplotlib.pyplot as plt

# === Your custom preprocessing classes ===

class Remove_correlateds:
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.is_trained = False
    def train(self, df_train):
        self.is_trained = True
        data = copy.deepcopy(df_train)
        corr_matrix = df_train.corr()
        corr_matrix = corr_matrix.abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        data.drop(self.to_drop, axis=1, inplace=True)
        return data
    def apply(self, df_test):
        data = copy.deepcopy(df_test)
        data.drop(self.to_drop, axis=1, inplace=True)
        return data

class Variance_threshold_selector:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.is_trained = False
    def train(self, df_train):
        self.is_trained = True
        data = copy.deepcopy(df_train)
        selector = VarianceThreshold(self.threshold)
        selector.fit(data)
        self.to_keep = selector.get_support(indices=True)
        data = data[data.columns[self.to_keep]]
        return data
    def apply(self, df_test):
        data = copy.deepcopy(df_test)
        data = data[data.columns[self.to_keep]]
        return data

class Normalizer:
    def __init__(self):
        self.is_trained = False
    def train(self, df_train):
        self.is_trained = True
        data = copy.deepcopy(df_train)
        columns = data.columns
        inds = data.index
        x = data.values
        self.min_max_scaler = MinMaxScaler()
        self.min_max_scaler.fit(x)
        x_scaled = self.min_max_scaler.transform(x)
        data = pd.DataFrame(x_scaled, columns=columns).set_index(inds)
        return data
    def apply(self, df_test):
        data = copy.deepcopy(df_test)
        columns = data.columns
        inds = data.index
        x = data.values
        x_scaled = self.min_max_scaler.transform(x)
        data = pd.DataFrame(x_scaled, columns=columns).set_index(inds)
        return data

class LightGBM_Pipeline():
    def __init__(self, best_params):
        self.is_trained = False
        self.best_params = best_params
    def train(self, df_train, y_train):
        self.is_trained = True
        self.columns = df_train.columns
        self.clf = lgb.LGBMClassifier(**self.best_params)
        self.clf.fit(df_train, y_train)
    def apply(self, df_test):
        if isinstance(df_test, (np.ndarray, np.generic)):
            df_test = pd.DataFrame(df_test, columns=self.columns)
        return self.clf.predict(df_test)
    def predict_proba(self, df_test):
        if isinstance(df_test, (np.ndarray, np.generic)):
            df_test = pd.DataFrame(df_test, columns=self.columns)
        return self.clf.predict_proba(df_test)[:, 1]

# === Seeding ===
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PCA_Custom:
    def __init__(self, var_threshold=0.95):
        self.var_threshold = var_threshold
        self.is_trained = False

    def fit(self, X):
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort by descending eigenvalue
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_idx]
        self.eigenvectors_ = eigenvectors[:, sorted_idx]  # full eigenvectors sorted

        # Step 5: Determine number of components for threshold
        total_variance = np.sum(self.eigenvalues_)
        var_ratio = np.cumsum(self.eigenvalues_) / total_variance
        self.num_components_ = np.searchsorted(var_ratio, self.var_threshold) + 1

        # Step 6: Save selected projection matrix
        self.components_ = self.eigenvectors_[:, :self.num_components_]
        self.is_trained = True

    def apply(self, X):
        if not self.is_trained:
            raise RuntimeError("PCA_Custom must be fitted first.")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)


random_seed(0, True)

# === Setup logging ===
log_filename = "single_split_lgbm_log.txt"
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# === Load dataset ===
logging.info("\nLoading PI-CAI dataset...")
data = pd.read_csv("/AI_Data/radMLBench/PI-CAI.csv")
logging.info(f"Loaded PI-CAI data shape: {data.shape}")

# Features and Labels
X = data.drop(columns=['ID', 'Target'])
y = data['Target'].values

# === Split into 60/20/20 ===
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=0, stratify=y_dev)

logging.info(f"Train shape: {X_train.shape}")
logging.info(f"Val shape: {X_val.shape}")
logging.info(f"Test shape: {X_test.shape}")

# === Preprocessing ===
remove_corr = Remove_correlateds()
X_train_corrremoved = remove_corr.train(X_train)
X_val_corrremoved = remove_corr.apply(X_val)
X_test_corrremoved = remove_corr.apply(X_test)

var_selector = Variance_threshold_selector()
X_train_varfiltered = var_selector.train(X_train_corrremoved)
X_val_varfiltered = var_selector.apply(X_val_corrremoved)
X_test_varfiltered = var_selector.apply(X_test_corrremoved)

normalizer = Normalizer()
X_train_normalized = normalizer.train(X_train_varfiltered)
X_val_normalized = normalizer.apply(X_val_varfiltered)
X_test_normalized = normalizer.apply(X_test_varfiltered)

# PCA
# === Fit PCA on DataFrame directly ===
pca = PCA_Custom(var_threshold=0.95)
pca.fit(X_train_normalized.values)  # Still use `.values` for fitting

# === Apply PCA and preserve DataFrame structure ===
X_train = pd.DataFrame(pca.apply(X_train_normalized.values), index=X_train.index)
X_val = pd.DataFrame(pca.apply(X_val_normalized.values), index=X_val.index)
X_test = pd.DataFrame(pca.apply(X_test_normalized.values), index=X_test.index)

# Optional: Rename columns for clarity
X_train.columns = [f"PC{i+1}" for i in range(X_train.shape[1])]
X_val.columns = X_train.columns
X_test.columns = X_train.columns


#Grid search
# === Manual Grid Search for LightGBM ===
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [30, 60]
}

# Generate all combinations
keys = list(param_grid.keys())
values = list(param_grid.values())

best_auc = -np.inf
best_params = None

logging.info("\nStarting manual grid search...")

for param_values in product(*values):
    params = dict(zip(keys, param_values))
    params['random_state'] = 0  # Always fix random_state

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)

    logging.info(f"Params {params} → Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        best_params = params

logging.info(f"\n Best Hyperparameters: {best_params} with Val AUC: {best_auc:.4f}")



# === Train LightGBM ===
pipeline = LightGBM_Pipeline(best_params)
pipeline.train(X_train, y_train)

# === Evaluate ===
y_pred_test = pipeline.apply(X_test)
y_pred_proba_test = pipeline.predict_proba(X_test)

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_test),
    'AUC': roc_auc_score(y_test, y_pred_proba_test),
    'F1-score': f1_score(y_test, y_pred_test),
    'Precision': precision_score(y_test, y_pred_test),
    'Recall': recall_score(y_test, y_pred_test),
}

logging.info("\n=== Test Set Metrics ===")
for k, v in metrics.items():
    logging.info(f"{k}: {v:.4f}")

# === Feature Importance (Normalized) ===

# Extract gain importances and normalize
gain_importance = pipeline.clf.booster_.feature_importance(importance_type='gain')
feature_names = X_test.columns
normalized_gain = gain_importance / np.sum(gain_importance)

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Normalized Gain Importance': normalized_gain
}).sort_values('Normalized Gain Importance', ascending=False)

# Save and log
feature_importance_df.to_csv('lgbm_gain_importance_normalized.csv', index=False)
logging.info("\nTop Normalized LightGBM Gain Importances:")
logging.info(feature_importance_df.head(10))


from sklearn.neural_network import MLPRegressor

# === Train MLP to learn mapping from PCA (125D) to full PCA projection (944D) ===
X_input = X_train.values  # shape: (n_samples, 125)
Y_target = np.dot(X_train_normalized, pca.eigenvectors_)  # shape: (n_samples, 944)

mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=1000, random_state=0)
mlp.fit(X_input, Y_target)

# === Use MLP to map gain importance vector to full PCA projection space ===
gain_importance_vector = normalized_gain.reshape(1, -1)  # shape (1, 125)
gain_importance_expanded = mlp.predict(gain_importance_vector)  # shape (1, 944)

# Step: Compare original and predicted PC importances for first pca.num_components_
mlp_estimate_subset = gain_importance_expanded[0][:pca.num_components_]
original_subset = gain_importance_vector[0]

# Print differences
diff_vector = original_subset - mlp_estimate_subset
diff_norm = np.linalg.norm(diff_vector)
print(f"🔍 Norm of difference in first {pca.num_components_} PCs: {diff_norm:.6f}")

# Step: Overwrite first pca.num_components_ values in gain_importance_expanded
gain_importance_expanded[0][:pca.num_components_] = gain_importance_vector[0]


# === Project MLP-expanded PCA importances back to original feature space ===
projected_feature_importance_vector = np.dot(gain_importance_expanded, pca.eigenvectors_.T).flatten()

# === Package into DataFrame and plot ===
original_feature_names = X_train_normalized.columns
mlp_projected_df = pd.DataFrame({
    'Feature': original_feature_names,
    'MLP Projected Gain Importance': np.abs(projected_feature_importance_vector)
}).sort_values('MLP Projected Gain Importance', ascending=False)

# Save and plot
mlp_projected_df.to_csv("lgbm_gain_importance_projected_mlp.csv", index=False)


# === (a) Top 5 Normalized LightGBM Features ===
fig, ax1 = plt.subplots(figsize=(10, 6))

top_features_a = feature_importance_df.head(5)
ax1.barh(
    top_features_a['Feature'],
    top_features_a['Normalized Gain Importance'],
    color='black',
    edgecolor='none'
)

# Remove chartjunk
for spine in ['top', 'right', 'left']:
    ax1.spines[spine].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('none')
ax1.grid(False)

ax1.tick_params(axis='both', which='major', labelsize=13)
ax1.set_xlabel("Normalized Gain Importance", fontsize=14)
ax1.set_ylabel("Feature", fontsize=14)
ax1.set_title("(a) Top 5 Normalized LightGBM Features", fontsize=15, pad=15)

plt.tight_layout()
plt.savefig("top5_normalized_lightgbm_features.png", dpi=300, bbox_inches='tight')


# === (b) Top 5 Projected Original Features (MLP) ===
fig, ax2 = plt.subplots(figsize=(10, 6))

top_features_b = mlp_projected_df.head(5)
ax2.barh(
    top_features_b['Feature'],
    top_features_b['MLP Projected Gain Importance'],
    color='black',
    edgecolor='none'
)

# Remove chartjunk
for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('none')
ax2.grid(False)

ax2.tick_params(axis='both', which='major', labelsize=13)
ax2.set_xlabel("Projected Gain Importance", fontsize=14)
ax2.set_ylabel("Feature", fontsize=14)
ax2.set_title("(b) Top 5 Projected Original Features (MLP)", fontsize=15, pad=15)

plt.tight_layout()
plt.savefig("top5_projected_mlp_features.png", dpi=300, bbox_inches='tight')
