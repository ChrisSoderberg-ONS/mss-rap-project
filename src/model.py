from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def evaluate_preds(y_true, y_preds):
    accuracy = r2_score(y_true, y_preds)
    precision = mean_absolute_error(y_true, y_preds)
    recall = mean_squared_error(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2)}
    print(f"Acc: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    return metric_dict

def train_model(X_train, y_train):
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    return rf

def predict_and_evaluate(model, X_valid, y_valid):
    y_preds = model.predict(X_valid)
    return evaluate_preds(y_valid, y_preds)
