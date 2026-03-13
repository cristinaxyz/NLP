import torch 
from data import load_data, build_loaders
from evaluation import (train_model, evaluate_model, plot_cm, plot_learning_curves, get_misclassified_examples)
from models.LSTM import LSTMClassifier
from models.CNN import CNNClassifier

"""
def present_results(train_ds, dev_ds, test_ds, seed, model, model_name, plot_title, plot_file_name):
    y_test, y_pred = model(train_ds, dev_ds, test_ds, seed)
    acc, macro_f1, cm = evaluate_predictions(y_test, y_pred)
    print(model_name)
    print("Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("Confusion Matrix:", cm)
    print(f"Misclassified words: {get_misclassified_examples(y_test, y_pred, test_ds)}\n")
    plot_cm(cm, plot_title, plot_file_name)
"""

def train_and_report(model, model_name, train_loader, dev_loader, test_loader, test_ds, device, lr, max_epochs):
    print(f"\nTraining {model_name}...")

    hist = train_model(
        model,
        train_loader,
        dev_loader,
        device,
        lr=lr,
        max_epochs=max_epochs,
        patience=3,
    )

    results = evaluate_model(model, test_loader, device)

    print(f"\n{model_name} results")
    print("Accuracy:", results["acc"])
    print("Macro F1:", results["f1"])

    plot_cm(results["cm"], f"Confusion Matrix - {model_name}", f"cm_{model_name.lower()}.png")

    errors = get_misclassified_examples(
        results["y_true"],
        results["y_pred"],
        test_ds,
        num_examples=10,
    )

    print(f"\nSome {model_name} errors:")
    for err in errors:
        print(err)

    return hist, results


def main():
    seed = 42
    batch_size = 64
    max_length = 64
    embed_dim = 64
    hidden_dim = 64
    num_filters = 64
    dropout = 0.3
    lr = 1e-3
    max_epochs = 3

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_ds, dev_ds, test_ds = load_data(seed)

    print("Train size:", len(train_ds))
    print("Dev size:", len(dev_ds))
    print("Test size:", len(test_ds))

    train_loader, dev_loader, test_loader, vocab, pad_idx = build_loaders(
        train_ds,
        dev_ds,
        test_ds,
        batch_size=batch_size,
        max_length=max_length,
    )

    vocab_size = len(vocab)

    # CNN
    cnn_model = CNNClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_filters=num_filters,
        kernel_sizes=(3, 4, 5),
        dropout=dropout,
        pad_idx=pad_idx,
        num_classes=4,
    ).to(device)

    hist_cnn, cnn_results = train_and_report(
        cnn_model,
        "CNN",
        train_loader,
        dev_loader,
        test_loader,
        test_ds,
        device,
        lr,
        max_epochs,
    )

    # LSTM
    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=dropout,
        pad_idx=pad_idx,
        num_classes=4,
        bidirectional=False,
    ).to(device)

    hist_lstm, lstm_results = train_and_report(
        lstm_model,
        "LSTM",
        train_loader,
        dev_loader,
        test_loader,
        test_ds,
        device,
        lr,
        max_epochs,
    )

    # comparison plots
    plot_learning_curves(
        [
            {"name": "CNN", "hist": hist_cnn},
            {"name": "LSTM", "hist": hist_lstm},
        ],
        "val_loss",
        "Validation Loss",
        "Loss",
        "val_loss.png",
    )

    plot_learning_curves(
        [
            {"name": "CNN", "hist": hist_cnn},
            {"name": "LSTM", "hist": hist_lstm},
        ],
        "val_f1",
        "Validation Macro F1",
        "Macro F1",
        "val_f1.png",
    )

    # ablation
    print("\nAblation study: LSTM dropout")

    for dropout_value in [0.1, 0.3, 0.5]:
        print(f"\nTraining LSTM with dropout = {dropout_value}")

        ablation_model = LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout_value,
            pad_idx=pad_idx,
            num_classes=4,
            bidirectional=False,
        ).to(device)

        train_model(
            ablation_model,
            train_loader,
            dev_loader,
            device,
            lr=lr,
            max_epochs=max_epochs,
            patience=3,
        )

        ablation_results = evaluate_model(ablation_model, test_loader, device)

        print(
            f"dropout={dropout_value} | "
            f"Accuracy={ablation_results['acc']:.4f} | "
            f"Macro F1={ablation_results['f1']:.4f}"
        )


if __name__ == "__main__":
    main()

"""
    present_results(train_ds, dev_ds, test_ds, seed, model_logistic_reg, "Logistic Regression", "Confusion Matrix, model TF-IDF + Logistic Regression", "cm_reg.png")
    present_results(train_ds, dev_ds, test_ds, seed, model_linear_svm, "Linear SVM", "Confusion Matrix, model: TF-IDF + Linear SVM", "cm_svm.png")
    
    model_LSTM = LSTMClassifier

    ablation_param = [0.1, 0.3, 0.5]

    for param in ablation_param:
        model_LSTM = LSTMClassifier(dropout=param)

if __name__ == "__main__":
    main()
"""