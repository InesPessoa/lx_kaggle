#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:26:41 2019

@author: inespessoa
"""

import pandas as pd
import numpy as np
import math

def vector_parameters(x, y, z, name):
    norm = np.sqrt(x**2 + y**2 + z**2)
    alpha = math.acos(x/norm)
    beta = math.acos(y/norm)
    gama = math.acos(z/norm)
    return pd.Series({name + "_norm": norm,
                      name + "_alpha": alpha,
                      name + "_beta": beta,
                      name + "_gama": gama})

def orientation(q0, q1, q2, q3):
    alpha = math.atan(2*(q0*q1 + q2*q3)/(1-2*(q1**2 + q2**2)))
    beta = math.asin(2*(q0*q2 - q3*q1))
    gama = math.atan(2*(q0*q3 + q1*q2)/(1-2*(q2**2 + q3**2)))
    return pd.Series({"orientation_alpha": alpha,
                      "orientation_beta": beta,
                      "orientation_gama": gama})

def metrics(series):
    series_mean = np.mean(series)
    series_std = np.std(series)
    series = series[np.logical_and((series <= (series_mean + 3*series_std)), 
                                   (series >= (series_mean - 3*series_std)))]
    return pd.Series({"mean": np.mean(series),
                     "std": np.std(series),
                     "max": np.max(series),
                     "min": np.min(series)})
    
def calculate_metrics(series):
    diff_series = abs(np.diff(series)) #fazer tambem sem a diferenca absoluta
    sparameters = metrics(series)
    sdiff = metrics(diff_series)
    return pd.Series({"mean": sparameters["mean"],
                     "std": sparameters["std"],
                     "max": sparameters["max"],
                     "min": sparameters["min"],
                     "diff_mean": sdiff["mean"],
                     "diff_std": sdiff["std"],
                     "diff_max": sdiff["max"],
                     "diff_min": sdiff["min"]})

def get_metrics(df):
    series_id = df.series_id.values[0]
    columns = df.columns
    columns = columns[columns!="series_id"]
    final_serie = pd.Series()
    for column in columns:
        series = calculate_metrics(df[column].values)
        for i in range(0, len(series.index)):
            name = series.index[i] + "_" + column
            final_serie[name] = series.iloc[i]
    final_serie["series_id"] = series_id
    return final_serie

df_x_train = pd.read_csv(r"/home/inespessoa/lx_kaggle/career-con-2019/X_train.csv", index_col=0)
df_y_train = pd.read_csv(r"/home/inespessoa/lx_kaggle/career-con-2019/y_train.csv")
df_x_train.sort_values(by=["series_id", "measurement_number"], inplace=True)
df_x_train[['orientation_X', 'orientation_Y', 'orientation_Z',
       'orientation_W']].apply(lambda x: orientation(x.orientation_X,
        x.orientation_Y,
        x.orientation_Z,
        x.orientation_W), axis=1)
df_angular_velocity = df_x_train[['angular_velocity_X', 'angular_velocity_Y',
       'angular_velocity_Z']].apply(lambda x: orientation(x.angular_velocity_X,
        x.angular_velocity_Y,
        x.angular_velocity_Z, "angular_velocity"), axis=1)
df_linear_accelaration = df_x_train[['linear_accelaration_X', 'linear_accelaration_Y',
       'linear_accelaration_Z']].apply(lambda x: orientation(x.linear_accelaration_X,
        x.linear_accelaration_Y,
        x.linear_accelaration_Z, "linear_accelaration"), axis=1)
df_x_train_v2 = pd.concat([df_x_train, df_angular_velocity, df_linear_accelaration], axis=1)
df_x_train_v2.drop(columns=["measurement_number"], inplace=True)
df_x_train_v3 = df_x_train_v2.groupby("series_id").apply(get_metrics)