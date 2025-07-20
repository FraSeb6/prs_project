import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)


def evaluate_model(model, X_test, y_test):
    """Print evaluation metrics for the model."""
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, average='macro')
    recall = recall_score(y_true, y_pred_labels, average='macro')
    f1 = f1_score(y_true, y_pred_labels, average='macro')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_true, y_pred_labels))
