import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from scipy.stats import ttest_rel


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_confusion_matrix(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def aggregate_fold_metrics(fold_metrics_list):
    keys = fold_metrics_list[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in fold_metrics_list]
        result[key] = {
            "mean": np.mean(values),
            "std": np.std(values, ddof=1),
            "values": values,
        }
    return result


def compute_ttest(metrics_a, metrics_b):
    results = {}
    for key in metrics_a:
        vals_a = metrics_a[key]["values"]
        vals_b = metrics_b[key]["values"]
        t_stat, p_value = ttest_rel(vals_a, vals_b)
        results[key] = {"t_stat": t_stat, "p_value": p_value}
    return results
