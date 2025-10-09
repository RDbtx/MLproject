import numpy as np
import time
import thrember
import os
import shutil
import json


def subset_generation(x: np.ndarray, y: np.ndarray, subset_len: int, results_dir: str, scenario: str):
    """subset_generation processes the dataset removing unlabeled data. It
    can also be used to generate a smaller subset of the dataset by modifying the
    subset_len parameter.

    inputs:
    x: np.ndarray containing the samples and their features
    y: np.ndarray containing the samples and their labels
    subset_len: is an integer that specifies how many samples we want to keep
    results_dir: Path to the directory where to store the results
    scenario: name of the scenario, could be TRAINING or TESTING

    outputs:
    x_small, y_small: np.ndarray containing the subset samples, their features and their labels"""
    subset = min(subset_len, len(x))
    idx = np.random.choice(len(x), subset, replace=False)
    x_small = x[idx]
    y_small = y[idx]

    # remove unlabeled samples
    x_small, y_small = subset_analysis(x_small, y_small, results_dir, scenario)

    return x_small, y_small


def shape_fixer(y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    """Shape_fixer function checks whether the number of feature inside
       the training set is the same as the number of features in the test set.
       If this is not the case, it changes the shape of the test set to contain only
       samples that the model was trained on.

       inputs:
       y_train: np.ndarray containing the training labels
       x_test: np.ndarray containing the test sample and features
       y_test: np.ndarray containing the test labels

       outputs:
       x_test: np.ndarray containing the corrected test sample and features
       y_test: np.ndarray containing the corrected tests samples and labels
       """

    if y_train.shape[1] != y_test.shape[1]:
        print("\nWARNING: y_train and y_test have different number of classes")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print("correcting y_train and y_test shape...")

        label_to_remove = y_test.shape[1] - 1
        mask_extra_label = y_test[:, label_to_remove] == 1
        n_extra = np.sum(mask_extra_label)

        x_test = x_test[~mask_extra_label]
        y_test = y_test[~mask_extra_label, :]
        y_test = np.delete(y_test, label_to_remove, axis=1)

        print(f"extra label samples: {n_extra}")
        print(f"corrected y_test shape: {y_test.shape}")
        print(f"corrected x_test shape: {x_test.shape}")

    return x_test, y_test


def subset_analysis(x_set: np.ndarray, y_set: np.ndarray, results_dir: str, scenario: str):
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
       """
    print(f"\n----{scenario} SUBSET ANALYSIS----")

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

    samples = 0
    for i in range(num_labels):
        samples_per_label = int(y_labeled[:, i].sum())
        print(f"Label {i}: {samples_per_label} samples")
        samples += samples_per_label

    print(f"\nTotal labeled samples counted: {samples}")

    # save dataset
    np.save(results_dir + f"x_{scenario}_small.npy", x_labeled)
    np.save(results_dir + f"y_{scenario}_small.npy", y_labeled)

    return x_labeled, y_labeled
