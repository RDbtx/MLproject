import json
import numpy as np
from sklearn.model_selection import train_test_split

SUBSET_DIR = "../Results/Subsets/"


def align_labels_columns(y_test: np.ndarray, test_labels: list, train_labels: list):
    """
    Pads and reorders y_test so that its columns match y_train exactly.
    Adds all-zero columns for missing classes and reorders existing ones to obtain corresponding labels.
    Returns the aligned y_test and the new label list.

    input:
    - y_test: np.ndarray containing test samples and their labels
    - test_labels: list containing all the labels of the test set
    - train_labels: list containing all the labels of the training set

    output:
    - y_test_aligned: np.ndarray of shape (samples,classes) corresponding to the y_train numpy array.
    - test_labels_aligned: list containing all the labels of the test set aligned with those of the training set.

    """

    label_idx_dict = {label: i for i, label in enumerate(test_labels)}

    aligned_cols = []
    for label in train_labels:
        if label in label_idx_dict:
            aligned_cols.append(y_test[:, label_idx_dict[label]])
        else:
            aligned_cols.append(np.zeros((y_test.shape[0],), dtype=y_test.dtype))

    y_test_aligned = np.stack(aligned_cols, axis=1)
    test_labels_aligned = train_labels[:]

    return y_test_aligned, test_labels_aligned


def set_generation(x: np.ndarray, y: np.ndarray, subset_len: int, scenario: str):
    """subset_generation processes the dataset removing unlabeled data. It
    can also be used to generate a smaller subset of the dataset by modifying the
    subset_len parameter.

    inputs:
    - x: np.ndarray containing the samples and their features
    - y: np.ndarray containing the samples and their labels
    - subset_len: is an integer that specifies how many samples we want to keep
    - scenario: name of the scenario, could be TRAINING or TESTING

    outputs:
    - label_names: list containing all the labels of the set
    - x_small, y_small: np.ndarray containing the subset samples, their features and their labels

    """

    subset = min(subset_len, len(x))
    idx = np.random.choice(len(x), subset, replace=False)
    x_small = x[idx]
    y_small = y[idx]

    # remove unlabeled samples
    labels_names, x_small, y_small = subset_labeling(x_small, y_small, scenario)

    return labels_names, x_small, y_small


def shape_fixer(min_samples_per_train_label: int, train_name: list, test_name: list,
                x_train: np.ndarray, y_train: np.ndarray,
                x_test: np.ndarray, y_test: np.ndarray):
    """
    This function modifies the shape of the training and test sets by removing unproperly labeled samples from both sets.
    It also removes all labels with fewer than min_samples_per_train_label samples.
    Subsequently, it aligns both train and test sets to have the same shape and labeling order and saves
    the new generated train and test sets.

    inputs:
    - min_samples_per_train_label: minimum number of samples per label
    - train_name: list containing all the labels of the training set
    - test_name: list containing all the labels of the test set
    - x_train: np.ndarray containing the train set's samples and their features
    - y_train: np.ndarray containing the train set's samples and their labels
    - x_test: np.ndarray containing the test set's samples and their features
    - y_test: np.ndarray containing the test set's samples and their labels

    outputs:
    - train_name: list containing all the labels of the training set
    - test_name: list containing all the labels of the test set
    - x_train: np.ndarray containing the newly generated train set's samples and their features
    - y_train: np.ndarray containing the newly generated train set's samples and their labels
    - x_test: np.ndarray containing the newly generated test set's samples and their features
    - y_test: np.ndarray containing the newly generated test set's samples and their labels

    """

    train_class_to_remove = [elem for elem in train_name if isinstance(elem, int)]
    num_labels = y_train.shape[1]
    for i in range(num_labels):
        samples_per_label = int(y_train[:, i].sum())
        if samples_per_label < min_samples_per_train_label:
            train_class_to_remove.append(train_name[i])

    test_class_to_remove = [
        elem for elem in test_name
        if isinstance(elem, int) or elem not in train_name or elem in train_class_to_remove
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

    # uncomment this if you want to save the newly generated sets

    # saving_time = time.time()
    # print("Saving new datasets...")
    # os.makedirs(SUBSET_DIR, exist_ok=True)
    # np.save(SUBSET_DIR + "x_train_small.npy", x_train)
    # print(f"x_train set saved at {SUBSET_DIR} x_train_small.npy")
    # np.save(SUBSET_DIR + "y_train_small.npy", y_train)
    # print(f"y_train set saved at {SUBSET_DIR} y_train_small.npy")
    # np.save(SUBSET_DIR + "x_test_small.npy", x_test)
    # print(f"x_test set saved at {SUBSET_DIR} x_test_small.npy")
    # np.save(SUBSET_DIR + "y_test_small.npy", y_test)
    # print(f"y_test set saved at {SUBSET_DIR} y_test_small.npy")
    # print(f"Saving time: {(time.time() - saving_time):.1f} s")
    # print(f"Dataset saved.")

    return train_name, test_name, x_train, y_train, x_test, y_test


def subset_labeling(x_set: np.ndarray, y_set: np.ndarray, scenario: str):
    """
    This function filters the datasets and attempts to label them with the dataset data before vectorization.
    It achieves this by identifying correspondences between pre-vectorization labels and their corresponding
    sample counts, and post-vectorization label indexes and their sample counts.
    If a clear correspondence is found, the index is labeled with its appropriate label name.
    Otherwise, it remains as a number, which is then removed by the shape fixer function.
    This is because we are not provided with a proper label vector, and the logic of vectorizing the test set
    and training set creates discrepancies in the indexes. It also removes all the unlabeled samples.

   inputs:
   - x_set: np.ndarray containing the set's samples and their features
   - y_set: np.ndarray containing the set's samples and their labels
   - scenario: str containing TRAINING if applied to the training set, TEST if applied to the test set.

   outputs:
   - label_name: list of label names
   - x_labeled: np.ndarray containing the newly labeled set's samples and their features
   - y_labeled: np.ndarray containing the newly labeled set's samples and their labels

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
        with open("../Behavior Dataset/Labels/train_result.json") as f:
            label_map = json.load(f)
            f.close()
    elif scenario == "TESTING":
        with open("../Behavior Dataset/Labels/test_result.json") as f:
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


def subset_analysis(x_set: np.ndarray, y_set: np.ndarray, scenario: str, labels_names=None) -> None:
    """
    subset_analysis analyzes the given set and prints on the screen all its relevant features,
    including the total number of samples, the shape of the feature and label sets, the number of labels,
    and a list of all the samples contained in each label.


    - x_set: np.ndarray containing the set's samples and their features
    - y_set:  np.ndarray containing the set's samples and their labels
    - scenario:  string containing TRAINING if applied to the training set, TEST if applied to the test set.
    - labels_names:  list of label names.

    """

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


def training_set_split(x: np.ndarray, y: np.ndarray, labels_names: list,
                       test_size: float = 0.33):
    """
    This function splits the input dataset into training and test sets.
    The split can be managed by modifying the test_size parameter, which can take values from 0.1 to 0.99.
    By default, the function splits the dataset into a 1/3 test set and a 2/3 training set.

    inputs:
    - x: np.ndarray containing the set's samples and their features
    - y: np.ndarray containing the set's samples and their labels
    - train_labels: list of label names.
    - test_size: size of the split between training and test sets.

    outputs:
    - x_train: np.ndarray containing the newly generated training set's samples and their features
    - y_train: np.ndarray containing the newly generated training set's samples and their labels
    - x_test: np.ndarray containing the newly generated test set's samples and their features
    - y_test: np.ndarray containing the newly generated test set's samples and their labels
    - test_labels: list of label names.

    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    test_labels = labels_names
    return x_train, x_test, y_train, y_test, test_labels
