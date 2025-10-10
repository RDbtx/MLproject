import time
import thrember
import os
import shutil
import json
import numpy as np

SUBSET_DIR = "../Results/Subsets/"


def align_labels_columns(y_test: np.ndarray, test_labels: list, train_labels: list):
    """
    Pads and reorders y_test so that its columns match train_labels exactly.
    Adds all-zero columns for missing classes and reorders existing ones.
    Returns the aligned y_test and the new label list (== train_labels).
    """
    # Create a map of current label → column index
    test_idx_of = {label: i for i, label in enumerate(test_labels)}

    # Build aligned matrix
    aligned_cols = []
    for label in train_labels:
        if label in test_idx_of:
            aligned_cols.append(y_test[:, test_idx_of[label]])  # existing column
        else:
            aligned_cols.append(np.zeros((y_test.shape[0],), dtype=y_test.dtype))  # new zero column

    # Stack all columns in the correct order
    y_test_aligned = np.stack(aligned_cols, axis=1)
    test_labels_aligned = train_labels[:]  # identical order and names

    return y_test_aligned, test_labels_aligned


def subset_generation(x: np.ndarray, y: np.ndarray, subset_len: int, scenario: str):
    """subset_generation processes the dataset removing unlabeled data. It
    can also be used to generate a smaller subset of the dataset by modifying the
    subset_len parameter.

    inputs:
    x: np.ndarray containing the samples and their features
    y: np.ndarray containing the samples and their labels
    subset_len: is an integer that specifies how many samples we want to keep
    scenario: name of the scenario, could be TRAINING or TESTING

    outputs:
    x_small, y_small: np.ndarray containing the subset samples, their features and their labels"""
    subset = min(subset_len, len(x))
    idx = np.random.choice(len(x), subset, replace=False)
    x_small = x[idx]
    y_small = y[idx]

    # remove unlabeled samples
    labels_names, x_small, y_small = subset_labeling(x_small, y_small, scenario)

    return labels_names, x_small, y_small


def shape_fixer(train_name: list, test_name: list,
                x_train: np.ndarray, y_train: np.ndarray,
                x_test: np.ndarray, y_test: np.ndarray):
    train_class_to_remove = [elem for elem in train_name if isinstance(elem, int)]
    test_class_to_remove = [
        elem for elem in test_name
        if isinstance(elem, int) or elem not in train_name
    ]

    print("\n\nWARNING! it seems like that labels are not properly set")
    print("Removing useless classes...")

    y_train_idx = np.argmax(y_train, axis=1)
    y_test_idx = np.argmax(y_test, axis=1)

    class_to_name = {i: name for i, name in enumerate(train_name)}

    train_remove_idx = [i for i, name in class_to_name.items() if name in train_class_to_remove]
    test_remove_idx = [i for i, name in class_to_name.items() if name in test_class_to_remove]

    if train_remove_idx:
        mask_train = ~np.isin(y_train_idx, train_remove_idx)
        x_train = x_train[mask_train]
        y_train = y_train[mask_train]

    if test_remove_idx:
        mask_test = ~np.isin(y_test_idx, test_remove_idx)
        x_test = x_test[mask_test]
        y_test = y_test[mask_test]

    train_drop_cols = [i for i, name in enumerate(train_name) if name in train_class_to_remove]
    test_drop_cols = [i for i, name in enumerate(test_name) if name in test_class_to_remove]

    if train_drop_cols:
        keep_cols_train = np.setdiff1d(np.arange(y_train.shape[1]), train_drop_cols, assume_unique=True)
        y_train = y_train[:, keep_cols_train]
        train_name = [train_name[i] for i in keep_cols_train]

    if test_drop_cols:
        keep_cols_test = np.setdiff1d(np.arange(y_test.shape[1]), test_drop_cols, assume_unique=True)
        y_test = y_test[:, keep_cols_test]
        test_name = [test_name[i] for i in keep_cols_test]

    train_name = [c for c in train_name if c not in train_class_to_remove]
    test_name = [c for c in test_name if c not in test_class_to_remove]

    print(f"Removed {len(train_class_to_remove)} classes from training set.")
    print(f"Removed {len(test_class_to_remove)} classes from test set.")
    print(f"Training samples left: {len(y_train)}")
    print(f"Testing samples left: {len(y_test)}")

    # here we align train and test set
    y_test, test_name = align_labels_columns(y_test, test_name, train_name)

    saving_time = time.time()
    print("Saving new datasets...")
    os.makedirs(SUBSET_DIR, exist_ok=True)
    np.save(SUBSET_DIR + "x_train_small.npy", x_train)
    print(f"x_train set saved at {SUBSET_DIR} x_train_small.npy")
    np.save(SUBSET_DIR + "y_train_small.npy", y_train)
    print(f"y_train set saved at {SUBSET_DIR} y_train_small.npy")
    np.save(SUBSET_DIR + "x_test_small.npy", x_test)
    print(f"x_test set saved at {SUBSET_DIR} x_test_small.npy")
    np.save(SUBSET_DIR + "y_test_small.npy", y_test)
    print(f"y_test set saved at {SUBSET_DIR} y_test_small.npy")
    print(f"Saving time: {(time.time() - saving_time):.1f} s")
    print(f"Dataset saved.")

    return train_name, test_name, x_train, y_train, x_test, y_test


def subset_labeling(x_set: np.ndarray, y_set: np.ndarray, scenario: str):
    """This functions filters the datasets and removes all the unlabeled samples.
       The subset analysis function was initially created just to have an idea of
       how many samples, features and labels where inside a given dataset and
       how many sample belonged to each label. But after noticing that the total
       number of samples was much higher than the total number of samples with labels,
       the function evolved allowing to remove the unlabeled samples from the dataset.

       inputs:
       x_set: np.ndarray (n_samples, n_features)
       y_set: np.ndarray (n_samples, n_labels)
       results_dir: Path to the directory where to store the results
       scenario: str containing TRAINING if applied to the training set, TEST if applied to the test set.

       outputs:
       label_name: list of label names
       x_small, y_small: np.ndarray containing the subset samples, their features and their labels
       """
    print(f"\n----{scenario} SUBSET LABELING----")

    labeled_mask = (y_set > 0).any(axis=1)

    x_labeled = x_set[labeled_mask]
    y_labeled = y_set[labeled_mask]

    print(f"Total samples:       {len(y_set)}")
    print(f"Labeled samples:     {len(y_labeled)}")
    print(f"Unlabeled removed:   {len(y_set) - len(y_labeled)}")

    print(f"\nSubset features shape: {x_labeled.shape}")
    print(f"Subset labels shape:   {y_labeled.shape}")

    num_labels = y_labeled.shape[1]
    print(f"\nThe dataset has {num_labels} label columns.")

    if scenario == "TRAINING":
        with open("../Dataset/Labels/train_result.json") as f:
            label_map = json.load(f)
            f.close()
    elif scenario == "TESTING":
        with open("../Dataset/Labels/test_result.json") as f:
            label_map = json.load(f)
            f.close()
    else:
        label_map = {}

    labels_names = []
    samples = 0
    for i in range(num_labels):
        samples_per_label = int(y_labeled[:, i].sum())

        matched_label = None
        for label, val in label_map.items():
            if label_map is not None and val == samples_per_label:
                if label not in labels_names and list(label_map.values()).count(val) == 1:
                    matched_label = label
                    break

        if matched_label:
            labels_names.append(matched_label)
        else:
            labels_names.append(i)

        print(f"Label {labels_names[i]}: {samples_per_label} samples")
        samples += samples_per_label

    print(f"\nTotal labeled samples counted: {samples}")

    return labels_names, x_labeled, y_labeled


def subset_analysis(x_set: np.ndarray, y_set: np.ndarray, scenario: str, labels_names=None):
    print(f"\n----{scenario} SUBSET ANALYSIS----\n")
    print(f"Total samples:       {len(x_set)}")
    print(f"\nSubset features shape: {x_set.shape}")
    print(f"Subset labels shape:   {y_set.shape}")
    num_labels = y_set.shape[1]
    print(f"\nThe dataset has {num_labels} label columns.")
    samples = 0
    for i in range(num_labels):
        samples_per_label = int(y_set[:, i].sum())
        if labels_names is not None:
            print(f"Label {labels_names[i]}: {samples_per_label} samples")
        else:
            print(f"Label {i}: {samples_per_label} samples")
        samples += samples_per_label

    print(f"\nTotal labeled samples counted: {samples}")
