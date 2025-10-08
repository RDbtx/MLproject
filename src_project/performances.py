from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import math


# =========================================================
# --- Helper plot functions ---
# =========================================================
def subset_analysis(x_set, y_set) -> None:
    print("\n\n----SUBSET ANALYSIS----")
    print(f"subset samples and features: {x_set.shape}")
    print(f"subet samples and labels:    {y_set.shape}")

    num_labels = y_set.shape[1]
    print(f"\nThe dataset has {num_labels} labels (columns).")
    for i in range(num_labels):
        samples_per_label = int(y_set[:, i].sum())
        print(f"Label {i + 1}: {samples_per_label}  samples")
    print("\n")


def evaluate_model(y_true, y_proba, threshold: float):
    """
        Compute precision, recall, and F1 for binary or multi-label problems.
        """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    if y_true.ndim == 1:
        # ---- Single-label ----
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
    else:
        # ---- Multi-label ----
        prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
        rec_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print("=== Micro-averaged metrics (overall) ===")
        print(f"Precision: {prec_micro:.4f}")
        print(f"Recall:    {rec_micro:.4f}")
        print(f"F1-score:  {f1_micro:.4f}")
        print("\n=== Macro-averaged metrics (per-label average) ===")
        print(f"Precision: {prec_macro:.4f}")
        print(f"Recall:    {rec_macro:.4f}")
        print(f"F1-score:  {f1_macro:.4f}")


def plot_roc(y_true, proba, label_names=None):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    if y_true.ndim == 1:
        # --- Single-label ---
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate");
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # --- Multi-label (plot ALL labels) ---
    L = y_true.shape[1]
    plt.figure(figsize=(9, 8))
    aucs = []
    for k in range(L):
        yk, pk = y_true[:, k], proba[:, k]
        # skip labels with a single class in ground truth
        if len(np.unique(yk)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yk, pk)
        auc = roc_auc_score(yk, pk)
        aucs.append(auc)
        name = label_names[k] if (label_names and k < len(label_names)) else f"label {k}"
        plt.plot(fpr, tpr, linewidth=1.0, alpha=0.8, label=f"{name} (AUC={auc:.3f})")

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), proba.ravel())
    auc_micro = roc_auc_score(y_true.ravel(), proba.ravel())
    plt.plot(fpr_micro, tpr_micro, linestyle="--", color="black", linewidth=2,
             label=f"micro (AUC={auc_micro:.3f})")

    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("False Positive Rate");
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves (all labels)")
    # Put legend outside to fit all entries
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    if aucs:
        print(f"Mean per-label AUC (plotted): {np.mean(aucs):.4f}")


def plot_conf_matrix(y_true, proba, threshold=0.2, label_index=None, label_names=None, cols=6):
    """
    - Single-label: standard CM.
    - Multi-label:
        * If label_index is not None -> CM for that label.
        * Else -> grid of per-label CMs (ALL labels).
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    if y_true.ndim == 1:
        # --- Single-label ---
        y_pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(cmap="Blues", ax=ax, values_format=".0f")
        ax.set_title(f"Confusion Matrix (thr={threshold})")
        plt.tight_layout();
        plt.show()
        return

    # --- Multi-label ---
    L = y_true.shape[1]

    if label_index is not None:
        # One specific label
        yk, pk = y_true[:, label_index], proba[:, label_index]
        y_pred = (pk >= threshold).astype(int)
        cm = confusion_matrix(yk, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        name = (label_names[label_index] if label_names and label_index < len(label_names)
                else f"label {label_index}")
        ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(cmap="Blues", ax=ax, values_format=".0f")
        ax.set_title(f"CM — {name} (thr={threshold})")
        plt.tight_layout();
        plt.show()
        return

    # Grid for ALL labels
    rows = math.ceil(L / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.4))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if k < L:
                yk, pk = y_true[:, k], proba[:, k]
                # handle degenerate case with one class present
                if len(np.unique(yk)) < 2:
                    ax.axis("off")
                    ax.set_title(
                        (label_names[k] if label_names and k < len(label_names) else f"label {k}") + " (skipped)")
                else:
                    y_pred = (pk >= threshold).astype(int)
                    cm = confusion_matrix(yk, y_pred, labels=[0, 1])
                    name = label_names[k] if (label_names and k < len(label_names)) else f"label {k}"
                    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
                    disp.plot(cmap="Blues", ax=ax, values_format=".0f", colorbar=False)
                    ax.set_title(f"{name}")
            else:
                ax.axis("off")
            k += 1

    plt.suptitle(f"Confusion Matrices — All Labels (thr={threshold})", y=1.02)
    plt.tight_layout()
    plt.show()
