#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ernest Namdar
"""

# Install the package if not already installed
# pip install radMLBench lightgbm scikit-learn scipy

import os
import pandas as pd
import lightgbm as lgb
# import radMLBench
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.decomposition import PCA
import numpy as np
import torch
import random
import logging
from scipy import stats

# === Seeding (to get reproducible results) ===
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # numpy seed
    torch.manual_seed(seed_value)  # torch CPU vars
    random.seed(seed_value)  # Python seed
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # torch GPU vars
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

random_seed(0, True)


log_filename = "10fold_cv_log.txt"
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


# === Load PI-CAI dataset ===
# logging.info("Available datasets:")
# for dataset in radMLBench.listDatasets():
#     logging.info(dataset)

logging.info("\nLoading PI-CAI dataset...")
data = radMLBench.loadData('PI-CAI')
logging.info(f"Loaded PI-CAI data shape: {data.shape}")

# Features and Labels
X = data.drop(columns=['ID', 'Target'])
y = data['Target'].values

logging.info(f"Feature matrix shape: {X.shape}")
logging.info(f"Labels shape: {y.shape}")
logging.info(f"Label distribution:\n{pd.Series(y).value_counts()}")
logging.info("\nStarting 10-fold cross-validation...")


# === LightGBM parameters ===
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 30
}

# === Collect per-fold metrics ===
acc_scores = []
auc_scores = []
f1_scores = []
precision_scores = []
recall_scores = []

# Stratified K-Fold Split
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
splits = list(kf.split(X, y))  # Materialize splits first

for fold_number, (train_index, test_index) in enumerate(splits, start=1):
    logging.info(f"Working on fold #{fold_number}/10")

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # === PCA: Fit on training set only ===
    pca = PCA(n_components=0.95, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train_pca, y_train)

    y_pred_fold = model.predict(X_test_pca)
    y_pred_prob_fold = model.predict_proba(X_test_pca)[:, 1]

    # Save per-fold metrics
    acc = accuracy_score(y_test, y_pred_fold)
    auc = roc_auc_score(y_test, y_pred_prob_fold)
    f1 = f1_score(y_test, y_pred_fold)
    precision = precision_score(y_test, y_pred_fold)
    recall = recall_score(y_test, y_pred_fold)

    acc_scores.append(acc)
    auc_scores.append(auc)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)

# === Final summary: Mean and 95% Confidence Interval ===
def mean_ci(scores):
    mean = np.mean(scores)
    ci95 = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=stats.sem(scores))
    lower_bound, upper_bound = ci95
    return mean, lower_bound, upper_bound

# Compute mean and CI for each metric
metrics_summary = {
    'Accuracy': mean_ci(acc_scores),
    'AUC': mean_ci(auc_scores),
    'F1-score': mean_ci(f1_scores),
    'Precision': mean_ci(precision_scores),
    'Recall': mean_ci(recall_scores),
}

# === Log final results ===
logging.info("\n=== Final 10-Fold CV Results with 95% Confidence Intervals ===")
for metric, (mean_value, ci_lower, ci_upper) in metrics_summary.items():
    logging.info(f"{metric}: {mean_value:.4f} ({ci_lower:.4f} - {ci_upper:.4f})")

logging.info("\nDone. See '10fold_cv_log.txt' for full progress and final metrics.")
