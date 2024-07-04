# tests/test_model.py
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from model import evaluate_preds, train_model, predict_and_evaluate

@pytest.fixture
def sample_model_df():
    np.random.seed(42)
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.rand(100)
    }
    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2']]
    y = df['target']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid

def test_evaluate_preds():
    y_true = np.array([3, -0.5, 2, 7])
    y_preds = np.array([2.5, 0.0, 2, 8])
    metrics = evaluate_preds(y_true, y_preds)
    assert metrics['accuracy'] == pytest.approx(0.94, 0.01)
    assert metrics['precision'] == pytest.approx(0.5, 0.01)
    assert metrics['recall'] == pytest.approx(0.375, 0.01)

def test_train_model(sample_model_df):
    X_train, X_valid, y_train, y_valid = sample_model_df
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 100  

def test_predict_and_evaluate(sample_model_df):
    X_train, X_valid, y_train, y_valid = sample_model_df
    model = train_model(X_train, y_train)
    metrics = predict_and_evaluate(model, X_valid, y_valid)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics