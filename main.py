from data import load_data
from evaluation import evaluate_predictions
from logistic_regression import model_logistic_reg

def main():
    train_ds, dev_ds, test_ds = load_data()
    print("Train size:")
    print(len(train_ds))
    print("Dev size:")
    print(len(dev_ds))
    print("Test size:")
    print(len(test_ds))

    y_test_reg, y_pred_reg = model_logistic_reg(train_ds, dev_ds, test_ds)

    acc_reg, macro_f1_reg, cm_reg = evaluate_predictions(y_test_reg, y_pred_reg)
    print("Accuracy:", acc_reg)
    print("Macro F1:", macro_f1_reg)
    print("Confusion Matrix:", cm_reg)

if __name__ == "__main__":
    main()
