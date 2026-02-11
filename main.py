from data import load_data
from evaluation import evaluate_predictions

def main():
    train_ds, dev_ds, test_ds = load_data()
    print("Train size:")
    print(len(train_ds))
    print("Dev size:")
    print(len(dev_ds))
    print("Test size:")
    print(len(test_ds))

if __name__ == "__main__":
    main()
