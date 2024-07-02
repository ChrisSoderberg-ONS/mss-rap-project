import numpy as np

def map_sex_column(df):
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    return df

def drop_top_25_height(df):
    height_75_percentile = df['Height'].quantile(0.75)
    df = df[df['Height'] < height_75_percentile]
    return df

def split_data(df):
    np.random.seed(42)
    df_shuffled = df.sample(frac=1)
    X = df_shuffled.drop("Rings", axis=1)
    y = df_shuffled["Rings"]
    train_split = round(0.7 * len(df_shuffled))
    valid_split = round(train_split + 0.15 * len(df_shuffled))
    X_train, y_train = X[:train_split], y[:train_split]
    X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
    X_test, y_test = X[valid_split:], y[valid_split:]
    return X_train, X_valid, X_test, y_train, y_valid, y_test
