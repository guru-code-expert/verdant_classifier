"""Evaluation and reporting utilities."""

import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def get_predictions_and_labels(model: tf.keras.Model, dataset: tf.data.Dataset):
    """Return true labels and predictions as numpy arrays."""
    all_preds = []
    all_labels = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(np.argmax(labels, axis=1) if labels.shape[-1] > 1 else labels.numpy())

    return np.array(all_labels), np.array(all_preds)

def print_evaluation_report(model, dataset, dataset_name="Test"):
    y_true, y_pred = get_predictions_and_labels(model, dataset)
    print(f"\nClassification Report ({dataset_name}):")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy ({dataset_name}): {accuracy_score(y_true, y_pred):.4f}\n")