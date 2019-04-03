#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:08:58 2019

@author: inespessoa
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn import feature_selection

df_train = pd.read_csv(r"train.csv", index_col=0)
df_x_test = pd.read_csv(r"test.csv", index_col=0)
labelencoder = LabelEncoder()
df_train["surface"] = labelencoder.fit_transform(df_train.surface.values)
clf_rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced")
clf_svm = SVC(kernel='poly', degree=2, C=0.2, class_weight='balanced')
fs_rf = SelectFromModel(clf_rf)
fs_svm = SelectFromModel(clf_svm)
fs = feature_selection.SelectPercentile(feature_selection.f_classif, percentile=10)
skf = StratifiedKFold(n_splits=5, random_state=True, shuffle=True)
pipeline = imbPipeline([
                        ("scale", StandardScaler()),
                        ("fs", fs),
                        ("reduce_dims", PCA(0.9)),
                        ("clf", clf_svm)
                        ])

y = df_train["group_id"].values
x = df_train.drop(columns=["group_id", "surface", "series_id"]).values
cm = []
score = []
for train_index, test_index in skf.split(x, y):
    df_train_split = df_train.iloc[train_index]
    x_train = df_train_split.drop(columns=["group_id", "surface", "series_id"]).values
    y_train = df_train_split["surface"].values
    df_test_split = df_train.iloc[test_index]
    x_test = df_test_split.drop(columns=["group_id", "surface", "series_id"]).values
    y_test = df_test_split["surface"].values
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    cm.append(confusion_matrix(y_test, y_pred))
    score.append(f1_score(y_test, y_pred, average='weighted'))

mean_cm = sum(cm)/len(cm)
mean_score = sum(score)/len(score)
y_train = df_train["surface"].values
df_train.drop(columns=["group_id", "surface", "series_id"], inplace=True)
x_train = df_train.values
pipeline.fit(x_train, y_train)
x_test = df_x_test[df_train.columns].values
series_id = df_x_test["series_id.1"].apply(lambda x: int(x)).values
y_pred = pipeline.predict(x_test)
y_pred_label = labelencoder.inverse_transform(y_pred)
df_results = pd.DataFrame({"series_id": series_id, "surface": y_pred_label})
df_results.set_index("series_id", inplace=True)
df_results.to_csv("results.csv")