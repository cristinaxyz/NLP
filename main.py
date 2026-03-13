import torch
import torch.nn as nn 
from data import load_data, build_loaders
from evaluation import (train_model, evaluate_model, plot_cm, plot_learning_curves, get_misclassified_examples, show_errors)
from models.LSTM import LSTMClassifier
from models.CNN import CNNClassifier
from pandas import DataFrame
import time 
import random 
import numpy as np


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
    
def set_seed(seed: int = 13) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_time(name, model, train_loader, dev_loader, test_loader, device, lr, max_epochs):
    print(f"\nTraining {name}...")

    t0 = time.perf_counter()

    hist = train_model(
        model,
        train_loader,
        dev_loader,
        device,
        lr=lr,
        max_epochs=max_epochs,
        patience=3,
    )

    total_time = time.perf_counter() - t0
    val = evaluate_model(model, dev_loader, device)
    test = evaluate_model(model, test_loader, device)

    return {
        "name": name,
        "hist": hist,
        "val": val,
        "test": test,
        "time_s_total": total_time,
    }


def main():
    seed = 13
    set_seed(seed)

    batch_size = 64
    max_length = 128
    embed_dim = 64
    hidden_dim = 128
    num_filters = 64
    dropout = 0.3
    lr = 1e-3
    max_epochs = 10

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

    #CNN model
    cnn_model = CNNClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_filters=num_filters,
        kernel_sizes=(3, 4, 5),
        dropout=dropout,
        pad_idx=pad_idx,
        num_classes=4,
    ).to(device)

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

    print("Number of trainable parameters:")
    print("LSTM:", count_parameters(lstm_model))
    print("CNN:", count_parameters(cnn_model))
    
    res_cnn = train_and_time(
        "CNN",
        cnn_model,
        train_loader,
        dev_loader,
        test_loader,
        device,
        lr,
        max_epochs,
    )

    res_lstm = train_and_time(
        "LSTM",
        lstm_model,
        train_loader,
        dev_loader,
        test_loader,
        device,
        lr,
        max_epochs,
    )

    rows = []
    for res in [res_cnn, res_lstm]: 
        rows.append([
            res["name"],
            res["val"]["acc"],
            res["val"]["f1"],
            res["test"]["acc"],
            res["test"]["f1"],
            res["time_s_total"],
        ])
    
    df_compare = DataFrame(
        rows,
       columns=["model", "val_acc", "val_macro_f1", "test_acc", "test_macro_f1", "train_time_s"],
    ).sort_values(by=["val_macro_f1", "val_acc"], ascending=False).reset_index(drop=True)
    print("\n Comparison table: ")
    print(df_compare)

    #confusion matrix
    plot_cm(res_cnn["test"]["cm"], "CNN Confusion Matrix", "cnn_cm.png")
    plot_cm(res_lstm["test"]["cm"], "LSTM Confusion Matrix", "lstm_cm.png")

    # comparison plots
    plot_learning_curves(
        [res_lstm, res_cnn],
        "val_loss",
        "Validation Loss",
        "Loss",
        "val_loss.png",
    )
    plot_learning_curves(
    [res_lstm, res_cnn],
    "train_loss",
    "Training Loss",
    "Loss",
    "train_loss.png",
    )


    plot_learning_curves(
        [res_lstm, res_cnn],
        "val_f1",
        "Validation Macro F1",
        "Macro F1",
        "val_f1.png",
    )
    errs_lstm = get_misclassified_examples(
        lstm_model,
        test_ds,
        vocab,
        max_length,
        device,
        max_items=10,
    )

    errs_cnn = get_misclassified_examples(
        cnn_model,
        test_ds,
        vocab,
        max_length,
        device,
        max_items=10,
    )

    show_errors("LSTM errors", errs_lstm[:10])
    print("\n" + "=" * 80 + "\n")
    show_errors("CNN errors", errs_cnn[:10])

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

        ablation_results = evaluate_model(ablation_model, dev_loader, device)

        print(
            f"dropout={dropout_value} | "
            f" Dev Accuracy={ablation_results['acc']:.4f} | "
            f" Dev Macro F1={ablation_results['f1']:.4f}"
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