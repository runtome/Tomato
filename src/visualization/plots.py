import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_loss_curves(train_losses, val_losses, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_probs, class_names, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    for i in range(num_classes):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Macro-average ROC
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    for i in fpr_dict:
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= len(fpr_dict)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(10, 8))

    for i in fpr_dict:
        plt.plot(fpr_dict[i], tpr_dict[i],
                 label=f"{class_names[i]} (AUC={roc_auc_dict[i]:.3f})")

    plt.plot(all_fpr, mean_tpr, linestyle="--", linewidth=2, color="black",
             label=f"Macro-average (AUC={macro_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
