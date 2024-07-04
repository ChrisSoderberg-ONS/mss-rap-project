"""
model.py

This module provides functions to train a RandomForestRegressor model, make predictions,
and evaluate the model's performance using various metrics.
"""

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def evaluate_preds(y_true, y_preds):
    """
    Evaluate the model predictions using R2 score, Mean Absolute Error, and Mean Squared Error.

    Args:
        y_true: True target values.
        y_preds: Predicted target values.

    Returns:
        dict: A dictionary containing the rounded values of accuracy (R2 score), precision (MAE), and recall (MSE).
    """
    accuracy = r2_score(y_true, y_preds)
    precision = mean_absolute_error(y_true, y_preds)
    recall = mean_squared_error(y_true, y_preds)
    
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2)}
    
    # Print the evaluation metrics
    print(f"Acc: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return metric_dict

def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model on the training data.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        RandomForestRegressor: The trained RandomForestRegressor model.
    """
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    return rf

def predict_and_evaluate(model, X_valid, y_valid):
    """
    Make predictions using the trained model and evaluate its performance.

    Args:
        model (RandomForestRegressor): The trained model.
        X_valid: Validation features.
        y_valid: True validation target values.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    y_preds = model.predict(X_valid)
    return evaluate_preds(y_valid, y_preds)


