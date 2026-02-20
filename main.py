from data import load_data
from evaluation import *
from logistic_regression import model_logistic_reg
from Linear_svm import model_linear_svm

def main():
    seed = 42
    train_ds, dev_ds, test_ds = load_data(seed)
    print("Train size:")
    print(len(train_ds))
    print("Dev size:")
    print(len(dev_ds))
    print("Test size:")
    print(len(test_ds))

    y_test_reg, y_pred_reg = model_logistic_reg(train_ds, dev_ds, test_ds, seed)
    y_test_svm, y_pred_svm = model_linear_svm(train_ds, dev_ds, test_ds, seed)

    acc_reg, macro_f1_reg, cm_reg = evaluate_predictions(y_test_reg, y_pred_reg)
    acc_svm, macro_f1_svm, cm_svm = evaluate_predictions(y_test_svm, y_pred_svm)

    print("Accuracy:", acc_reg)
    print("Macro F1:", macro_f1_reg)
    print("Confusion Matrix:", cm_reg)

    print("svm accuracy: ", acc_svm)
    print("svm macro f1: ", macro_f1_svm)
    print("confusion matrix: ", cm_svm)

    print(f"logistic regression misclassified words: {get_misclassified_examples(y_test_reg, y_pred_reg, test_ds)}\n")
    print(f"linear supervised learning misclassified words: {get_misclassified_examples(y_test_svm,y_pred_svm, test_ds)}")

    plot_cm(cm_reg, "Confusion Matrix, model TF-IDF + Logistic Regression", "cm_reg.png")
    plot_cm(cm_svm, "Confusion Matrix, model: TF-IDF + Linear SVM", "cm_svm.png")

if __name__ == "__main__":
    main()
