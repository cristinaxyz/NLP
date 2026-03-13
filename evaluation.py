from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn 
import time
from data import tokenize, numericalize, UNK, normalization


def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average = 'macro')
    cm = confusion_matrix(y_true, y_pred)
    return acc, macro_f1, cm

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech",}

def get_misclassified_examples(model: nn.Module, raw_split, vocab, max_length, device, max_items: int = 8):
    model.eval()
    errs = []
    for ex in raw_split:
        text = normalization["title"] + " " + normalization["description"]
        tokens = tokenize(text)
        ids = numericalize(tokens, vocab)[:max_length]
        if len(ids) == 0:
            ids = [vocab[UNK]]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
        y = int(ex["label"])

        with torch.no_grad():
            logits = model(x, lengths)
            pred = int(logits.argmax(dim=1).item())
        if pred != y:
            snippet = text.replace("\n", " ")
            snippet = snippet[:250] + ("..." if len(snippet) > 250 else "")
            errs.append((y, pred, snippet))
        if len(errs) >= max_items:
            break

    return errs

def show_errors(name: str, errs):
    print(name)
    for i,(y,p,snip) in enumerate(errs):
        print()
        print(f"error {i+1}")
        print("true:", LABELS[y], "pred:", LABELS[p])
        print("text:", snip)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    correct = 0

    for batch in loader:
        x = batch.x.to(device)
        lengths = batch.lengths.to(device)
        y = batch.y.to(device)

        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n = n + y.size(0)
        correct += (logits.argmax(dim=1) ==y).sum().item()
    
    train_loss = total_loss / max(1,n)
    train_acc = correct/max(1,n)
    return train_loss, train_acc


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            lengths = batch.lengths.to(device)
            y = batch.y.to(device)

            logits = model(x, lengths)
            loss = loss_fn(logits, y)

            pred = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            total_loss += loss.item() * y.size(0)
            n += y.size(0)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    acc, macro_f1, cm = evaluate_predictions(y_true, y_pred)

    return {
        "loss": total_loss / max(1, n),
        "acc": acc,
        "f1": macro_f1,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    lr=1e-3,
    max_epochs=10,
    weight_decay=0.0,
    patience=3,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0
    hist = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, device)

        dt = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_results["loss"],
            "val_acc": val_results["acc"],
            "val_f1": val_results["f1"],
            "time_s": dt,
        }
        hist.append(record)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_results['loss']:.4f} | "
            f"val acc {val_results['acc']:.4f} | "
            f"val f1 {val_results['f1']:.4f} | "
            f"time {dt:.1f}s"
        )

        if val_results["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_results["loss"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                print("Early stopping triggered, restoring best parameters.")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return hist

def evaluate_model(model, loader, device): 
    return evaluate(model, loader, device)

def plot_cm(cm, title, path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation = 'nearest')
    plt.title(title)
    plt.colorbar()
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_learning_curves(results, key: str, title: str, ylabel: str, path: str):
    plt.figure()

    for res in results:
        hist = res["hist"]
        epochs = [h["epoch"] for h in hist]
        vals = [h[key] for h in hist]
        plt.plot(epochs, vals, label=res["name"])

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    #in main/py
    #plot_learning_curves([res_lstm, res_cnn], "val_loss", "Validation loss", "loss")
    #plot_learning_curves([res_lstm, res_cnn], "val_f1", "Validation macro F1", "macro F1")