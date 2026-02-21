from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average = 'macro')
    cm = confusion_matrix(y_true, y_pred)
    return acc, macro_f1, cm

def get_misclassified_examples(y_test, y_pred, texts, num_examples=20):
    """
    Returns misclassified examples.
    """
    misclassified = []
    for text, true, pred in zip(texts, y_test, y_pred):
        if true != pred:
            readable_text = text["title"]

            misclassified.append({
                "text": readable_text,
                "true_label": true,
                "predicted_label": pred
            })
        if len(misclassified) >= num_examples:
            break
    return misclassified

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
    plt.show()
    plt.close()