import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def split_features_target(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler