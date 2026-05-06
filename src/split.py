from sklearn.model_selection import train_test_split

def split_data(df, target='SalePrice', test_size=0.2, random_state=42):
    if target not in df.columns:
        raise ValueError(f"{target} not in dataframe")

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
