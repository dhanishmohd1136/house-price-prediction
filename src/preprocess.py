import pandas as pd
import numpy as np

def preprocess(df):
    df = df.copy()

    # Drop ID
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Target transform (train only)
    if 'SalePrice' in df.columns:
        df['SalePrice'] = np.log1p(df['SalePrice'])

    # -------- Missing values --------
    # Categorical → "None"
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('None')

    # Numerical → median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # -------- Transformations --------
    if 'LotArea' in df.columns:
        df['LotArea'] = np.log1p(df['LotArea'])

    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)

    # Drop weak columns
    df = df.drop(columns=['Street', 'Utilities'], errors='ignore')

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    return df