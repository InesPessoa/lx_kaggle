#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:26:41 2019

@author: inespessoa
"""

import pandas as pd
import numpy as np

def vector_parameters(x, y, z, name):
    norm = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arccos(x/norm)
    beta = np.arccos(y/norm)
    gama = np.arccos(z/norm)
    return pd.DataFrame({name + "_norm": norm,
                      name + "_alpha": alpha,
                      name + "_beta": beta,
                      name + "_gama": gama})

def orientation(q0, q1, q2, q3):
    alpha_numerator = 2*(q0*q1 + q2*q3)
    alpha_denominator = 1-2*(q1**2 + q2**2)
    alpha = np.arctan2(alpha_numerator, alpha_denominator)
    inside_beta = 2*(q0*q2 - q3*q1)
    inside_beta[inside_beta > 1] = 1
    inside_beta[inside_beta < -1] = -1
    beta = np.arcsin(inside_beta)
    gama_numerator = 2*(q0*q3 + q1*q2)
    gama_denominator = 1-2*(q2**2 + q3**2)
    gama = np.arctan2(gama_numerator, gama_denominator)
    return pd.DataFrame({"orientation_alpha": alpha,
                      "orientation_beta": beta,
                      "orientation_gama": gama})

def metrics(series):
    series_mean = np.mean(series)
    series_std = np.std(series)
    series = series[np.logical_and((series <= (series_mean + 3*series_std)), 
                                   (series >= (series_mean - 3*series_std)))]
    return np.array([np.mean(series), np.std(series),
                    np.max(series), np.min(series)])
    
def calculate_metrics(series):   
    diff_series_metrics = metrics(abs(np.diff(series)))
    grad_series_metrics = metrics(abs(np.gradient(series)))
    series_metrics = metrics(series)
    return np.concatenate([diff_series_metrics,
                           grad_series_metrics,
                           series_metrics])

def get_metrics(df):
    columns = df.columns
    columns = columns[columns!="series_id"]
    dataframe = df[columns].apply(calculate_metrics)
    values = dataframe.values.reshape(1,
                                len(dataframe.index)*len(dataframe.columns))[0]
    series = pd.Series(data=values, index=np.arange(0, len(values)))
    series["series_id"] = df["series_id"].values[0]
    return series

def create_new_features(df):
    df.sort_values(by=["series_id", "measurement_number"], inplace=True)
    df.reset_index(inplace=True)
    df_orientation = orientation(df['orientation_X'].values,
                                 df['orientation_Y'].values, 
                                 df['orientation_Z'].values,
                                 df['orientation_W'].values)
    df_angular_velocity = vector_parameters(df['angular_velocity_X'].values,
                                df['angular_velocity_Y'].values,
                                df['angular_velocity_Z'].values,
                                "angular_velocity")
    df_linear_accelaration = vector_parameters(df['linear_acceleration_X'].values,
                                df['linear_acceleration_Y'].values,
                                df['linear_acceleration_Z'].values,
                                "linear_acceleration")
    df = pd.concat([df,
                    df_orientation, df_angular_velocity,
                    df_linear_accelaration], axis=1)
    df.set_index("row_id", inplace=True)
    df.drop(columns=["measurement_number"], inplace=True)
    df = df.groupby("series_id").apply(get_metrics)
    return df
    

df_x_train = pd.read_csv(r"/home/inespessoa/lx_kaggle/career-con-2019/X_train.csv", index_col=0)
df_x_test = pd.read_csv(r"/home/inespessoa/lx_kaggle/career-con-2019/X_test.csv", index_col=0)
df_y_train = pd.read_csv(r"/home/inespessoa/lx_kaggle/career-con-2019/y_train.csv")
df_x_train = create_new_features(df_x_train)
df_train = pd.merge(df_x_train, df_y_train, how="left", on="series_id")
df_x_test = create_new_features(df_x_test)
df_train.to_csv("train.csv")
df_x_test.to_csv("test.csv")