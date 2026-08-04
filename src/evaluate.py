from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_classifier(
    y_true,
    predictions,
    probabilities
):
    """
    Compute classification metrics.
    """

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions),
        "Recall": recall_score(y_true, predictions),
        "F1": f1_score(y_true, predictions),
        "ROC-AUC": roc_auc_score(y_true, probabilities)
    }


def compute_confusion_matrix(
    y_true,
    predictions
):
    """
    Return the confusion matrix.
    """
    return confusion_matrix(
        y_true,
        predictions
    )