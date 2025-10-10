from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, \
    precision_score, recall_score, hamming_loss,f1_score
import matplotlib.pyplot as plt
import numpy as np
import os

RF_RESULT_DIR = "../Results/decision_forest/"


# =====================================
# --- General section ---
# =====================================

def to_class_indices(y: np.ndarray) -> np.ndarray:
    """This utility function is used to convert sample,label np.ndarray
    into class indices for confusion matrix computation.

    inputs:
    y: np.ndarray containing the samples and their labels

    outputs:
    y: contains the 1D class indices
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1)
    return y


def model_performances_report_generation(accuracy, precision, recall, class_report, conf_matrix, scenario: str,
                                         output_dir: str):
    """This function is used to generate the performance report.

    inputs:
    accuracy: Accuracy of the model
    precision: Precision of the model
    recall: Recall of the model
    class_report: Class report of the model
    conf_matrix: Confusion matrix of the model
    scenario: Name of the scenario, it could be TRAINING or TESTING
    output_dir: Path to the output directory
    """
    report_dir = output_dir + "/reports/"
    os.makedirs(report_dir, exist_ok=True)

    print("Performance Report generation...")
    report_str = []
    report_str.append(f"\n----{scenario} PERFORMANCES----")
    report_str.append(f"Accuracy: {accuracy:.4f}")
    report_str.append(f"Precision: {precision:.4f}")
    report_str.append(f"Recall: {recall:.4f}")
    report_str.append("\nClassification Report:\n")
    report_str.append(class_report)

    report_str.append("\nConfusion Matrix:\n")
    report_str.append(np.array2string(conf_matrix))

    output = "\n".join(report_str)

    file_path = report_dir + scenario + "_performances_report.txt"
    with open(file_path, "w") as f:
        f.write(output)
    print(f"Results saved to {file_path}")


# =====================================
# --- RF section ---
# =====================================

def model_performances_multiclass(labels_names: list, y_true: np.ndarray, y_pred: np.ndarray, scenario: str):
    """This function is used to compute the performances of our model. It computes
       model accuracy, precision, recall, class report and confusion matrix for the given scenario
       which could be TRAINING or TESTING.

       inputs:
       y_true: np.ndarray containing the subset samples, and their true labels
       y_pred: np.ndarray containing the subset samples, and their predicted labels
       scenario: Name of the scenario, it could be TRAINING or TESTING

       outputs:
       accuracy: Accuracy of the model
       precision: Precision of the model
       recall: Recall of the model
       class_report: Class report of the model
       conf_matrix: Confusion matrix of the model
    """
    all_labels = np.arange(y_true.shape[1])
    samples = y_true.shape[0]
    labels = y_true.shape[1]
    y_true = to_class_indices(y_true)
    y_pred = to_class_indices(y_pred)

    precision = precision_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average="macro", zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    class_report = classification_report(y_true, y_pred, labels=all_labels, target_names=labels_names, zero_division=0)

    print(f"\n----{scenario} PERFORMANCES----")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    print("\nClassification Report:")
    print(class_report)

    fig, ax = plt.subplots(figsize=(28, 28))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)
    disp.plot(ax=ax, cmap='Blues', include_values=True, xticks_rotation=90)
    plt.title(f"{scenario} Confusion Matrix - {labels} Classes {samples} Samples", fontsize=20, pad=20)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RF_RESULT_DIR, scenario + "_conf_matrix.png"))
    plt.show()

    return accuracy, precision, recall, class_report, cm


# =====================================
# --- KNN section ---
# =====================================

def model_performances_multiclass_knn(labels_names: list, y_true: np.ndarray, y_pred: np.ndarray, scenario: str):
    """
    Faster multiclass metrics for kNN (or any classifier).
    Returns (accuracy, precision_macro, recall_macro, class_report(None), cm)
    and always plots the confusion matrix styled like your reference.
    """

    # Compute macro metrics
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    hamm_loss = hamming_loss(y_true, y_pred)
    class_report = None
    cm = None

    # Logging
    classes = y_true.shape[1]
    samples = y_true.shape[0]
    print(f"\n----{scenario} PERFORMANCES----")
    print(f"Samples: {samples} | Classes: {classes}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (macro)")
    print(f"Recall:    {recall:.4f} (macro)")
    print(f"F1-score:   {f1_micro:.4f} (micro)")
    print(f"F1-score:   {f1_macro:.4f} (macro)")
    print(f"Hamming Loss:  {hamm_loss:.4f} (macro)")


    return accuracy, precision, recall, class_report, cm
