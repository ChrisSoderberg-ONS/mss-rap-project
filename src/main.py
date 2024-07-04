"""
main.py

This script orchestrates the data loading, preprocessing, visualization, model training, and evaluation processes.
It imports necessary functions from other modules and runs the main workflow.
"""

import data_loading
import data_preprocessing
import plotting
import model

def main():
    """
    Main function to execute the data processing, visualization, and model training pipeline.
    """
    # Load and preprocess the data
    data = data_loading.load_data()
    data = data_preprocessing.map_sex_column(data)
    data = data_preprocessing.drop_top_25_height(data)
    
    # Visualize the data
    plotting.plot_scatter_matrix(data)
    plotting.plot_height_boxplot(data)

    # Split the data
    X_train, X_valid, X_test, y_train, y_valid, y_test = data_preprocessing.split_data(data)

    # Train the model
    rf_model = model.train_model(X_train, y_train)

    # Evaluate the model
    baseline_metrics = model.predict_and_evaluate(rf_model, X_valid, y_valid)
    print(baseline_metrics)

if __name__ == "__main__":
    main()
