from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, \
    precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import os

RF_RESULT_DIR = "../Results/decision_forest/"


def to_class_indices(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1)
    return y


def model_performances_report_generation(accuracy, precision, recall, class_report, conf_matrix, scenario: str,
                                         ouput_dir: str):
    report_dir = ouput_dir + "/reports/"
    os.makedirs(report_dir, exist_ok=True)

    print("Performance Report generation...")
    report_str = []
    report_str.append(f"\n----{scenario} PERFORMANCES----")
    report_str.append(f"Accuracy: {accuracy:.4f}")
    report_str.append(f"Precision: {precision:.4f}")
    report_str.append(f"Recall: {recall:.4f}")
    report_str.append("\nClassification Report:\n")
    report_str.append(class_report)

    # Confusion matrix
    report_str.append("\nConfusion Matrix:\n")
    report_str.append(np.array2string(conf_matrix))

    # Combine all text
    output = "\n".join(report_str)

    # Optionally save to a file
    file_path = report_dir + scenario + "_performances_report.txt"
    if file_path:
        with open(file_path, "w") as f:
            f.write(output)
        print(f"Results saved to {file_path}")


def model_performances_multiclass(y_true: np.ndarray, y_pred: np.ndarray, scenario: str):
    all_labels = np.arange(y_pred.shape[1])
    samples = y_true.shape[0]
    labels = y_true.shape[1]
    y_true = to_class_indices(y_true)
    y_pred = to_class_indices(y_pred)

    precision = precision_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average="macro", zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=all_labels, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    class_report = classification_report(y_true, y_pred, labels=all_labels, zero_division=0)

    print(f"\n----{scenario} PERFORMANCES----")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    print("\nClassification Report:")
    print(class_report)

    fig, ax = plt.subplots(figsize=(28, 28))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(ax=ax, cmap='Blues', include_values=True, xticks_rotation=90)

    # Improve readability
    plt.title(f"{scenario} Confusion Matrix - {labels} Classes {samples} Samples", fontsize=20, pad=20)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

    return accuracy, precision, recall, class_report, cm

# print("Loading data...")
# y_small = np.load(DIR + "y_small.npy")
# predictions = np.load(DIR + "train_prediction.npy")
# print("Data loaded.")

# accuracy, class_report, cm = model_performances_multiclass(y_small, predictions, scenario="TRAINING")
# model_performances_report_generation(accuracy, class_report, cm, "TRAINING")
