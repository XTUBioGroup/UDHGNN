# evaluate.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, auc


def evaluate_predictions(outputs, labels, idx_test):
    """
    Evaluate predictions
    """
    # Ensure outputs shape is correct
    if len(outputs.shape) > 1:
        outputs = outputs.squeeze()

    # Compute prediction probabilities
    probs = torch.sigmoid(outputs).cpu().detach().numpy()

    # Extract test set
    test_probs = probs[idx_test.cpu()]
    test_labels = labels[idx_test.cpu()].cpu().numpy()

    # Check number of positive samples
    pos_count = np.sum(test_labels)
    neg_count = len(test_labels) - pos_count

    if pos_count == 0 or neg_count == 0:
        print(f"Warning: Positive samples={pos_count}, Negative samples={neg_count} in test set")
        return 0.5, float(np.mean(test_labels)), 0.0

    # Compute AUROC
    try:
        auroc = roc_auc_score(test_labels, test_probs)
    except Exception as e:
        print(f"AUROC calculation failed: {e}")
        auroc = 0.5

    # Compute AUPRC
    try:

        auprc = average_precision_score(test_labels, test_probs)
    except Exception as e:
        print(f"AUPRC calculation failed: {e}")
        auprc = float(np.mean(test_labels))

    # Compute F1 Score
    try:
        # Use fixed threshold 0.5
        test_preds = (test_probs > 0.5).astype(int)
        f1 = f1_score(test_labels, test_preds, average='binary')
    except Exception as e:
        print(f"F1 calculation failed: {e}")
        f1 = 0.0

    return auroc, auprc, f1