from data import load_data
from evaluation import *
from models.logistic_regression import model_logistic_reg
from models.Linear_svm import model_linear_svm

def present_results(train_ds, dev_ds, test_ds, seed, model, model_name, plot_title, plot_file_name):
    y_test, y_pred = model(train_ds, dev_ds, test_ds, seed)
    acc, macro_f1, cm = evaluate_predictions(y_test, y_pred)
    print(model_name)
    print("Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("Confusion Matrix:", cm)
    print(f"Misclassified words: {get_misclassified_examples(y_test, y_pred, test_ds)}\n")
    plot_cm(cm, plot_title, plot_file_name)

def main():
    seed = 42
    train_ds, dev_ds, test_ds = load_data(seed)
    print("Train size:")
    print(len(train_ds))
    print("Dev size:")
    print(len(dev_ds))
    print("Test size:")
    print(len(test_ds))

    present_results(train_ds, dev_ds, test_ds, seed, model_logistic_reg, "Logistic Regression", "Confusion Matrix, model TF-IDF + Logistic Regression", "cm_reg.png")
    present_results(train_ds, dev_ds, test_ds, seed, model_linear_svm, "Linear SVM", "Confusion Matrix, model: TF-IDF + Linear SVM", "cm_svm.png")

if __name__ == "__main__":
    main()
