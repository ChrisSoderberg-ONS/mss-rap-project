import data_loading
import data_preprocessing
import plotting
import model

def main():
    data = data_loading.load_data()
    data = data_preprocessing.map_sex_column(data)
    data = data_preprocessing.drop_top_25_height(data)

    plotting.plot_scatter_matrix(data)
    plotting.plot_height_boxplot(data)

    X_train, X_valid, X_test, y_train, y_valid, y_test = data_preprocessing.split_data(data)

    rf_model = model.train_model(X_train, y_train)

    baseline_metrics = model.predict_and_evaluate(rf_model, X_valid, y_valid)
    print(baseline_metrics)

if __name__ == "__main__":
    main()
