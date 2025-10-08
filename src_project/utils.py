import numpy as np
import time
import thrember
import os
import shutil


def subset_generation(x: np.ndarray, y: np.ndarray, subset_len: int, results_dir: str, scenario: str):
    subset = min(subset_len, len(x))
    idx = np.random.choice(len(x), subset, replace=False)
    x_small = x[idx]
    y_small = y[idx]
    np.save(results_dir + f"x_{scenario}_small.npy", x_small)
    np.save(results_dir + f"y_{scenario}_small.npy", y_small)
    return x_small, y_small


def shape_fixer(labels_train: np.ndarray, labels_test: np.ndarray) -> None:
    if labels_train.shape[1] != labels_test.shape[1]:
        print("\nWARNING: y_train and y_test have different number of classes")
        print("correcting y_train and y_test shape...")
        labels_test = labels_test[:, :labels_train.shape[1]]
        print(f"y_train shape: {labels_train.shape}")
        print(f"y_test shape: {labels_test.shape}")


def subset_analysis(x_set: np.ndarray, y_set: np.ndarray, scenario: str):
    """This functions filters the datasets and removes all the unlabeled samples.
       The subset analysis function was initally created just to have an idea of
       how many samples, features and labels where inside a given dataset and
       how many sample belonged to each label. But after noticing that the total
       number of samples was much higher than the total number of samples with labels,
       the function evolved allowing to remove the unlabeled samples from the dataset.

       inputs:
       x_set: np.ndarray (n_samples, n_features)
       y_set: np.ndarray (n_samples, n_labels)
       scenario: str  (TRAINING if applied to the training set, TEST if applied to the test set ecc..)
       """
    print(f"\n----{scenario} SUBSET ANALYSIS----")

    if np.isnan(y_set).any():
        labeled_mask = ~np.isnan(y_set).all(axis=1)
    else:
        labeled_mask = (y_set.sum(axis=1) > 0)

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

    return x_labeled, y_labeled
