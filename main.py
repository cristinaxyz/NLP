from data import load_data
from evaluation import evaluate_predictions
from logistic_regression import model_logistic_reg
from Linear_svm import model_linear_svm

def main():
    train_ds, dev_ds, test_ds = load_data()
    print("Train size:")
    print(len(train_ds))
    print("Dev size:")
    print(len(dev_ds))
    print("Test size:")
    print(len(test_ds))

    y_test_reg, y_pred_reg = model_logistic_reg(train_ds, dev_ds, test_ds)
    y_test_svm, y_pred_svm = model_linear_svm(train_ds, dev_ds, test_ds)

    acc_reg, macro_f1_reg, cm_reg = evaluate_predictions(y_test_reg, y_pred_reg)
    acc_svm, macro_f1_svm, cm_svm = evaluate_predictions(y_test_svm, y_pred_svm)

    print("Accuracy:", acc_reg)
    print("Macro F1:", macro_f1_reg)
    print("Confusion Matrix:", cm_reg)

    print("svm accuracy: ", acc_svm)
    print("svm macro f1: ", macro_f1_svm)
    print("confusion matrix: ", cm_svm)

if __name__ == "__main__":
    main()
