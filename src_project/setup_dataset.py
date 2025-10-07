import thrember
import os


def download_dataset(dataset_path: str) -> None:
    """Download the dataset and extract it into the Dataset folder.
    Once the dataset .jsonl file are all unzipped, the feature extraction
    function is called to transform raw data into vectorized features.

    input:
    dataset_path: path to the dataset folder
    """
    # Dataset creation
    os.makedirs(dataset_path, exist_ok=True)

    # Dataset download
    thrember.download_dataset(dataset_path, file_type="PE")

    # vectorized features extraction
    feature_extraction(dataset_path)


def feature_extraction(dataset_path: str) -> None:
    """this function checks whether the dataset folder exists, and then calls the thrember
    create_vectorized_features function in order to extract features into vectorized features.

    input:
    dataset_path: path to the dataset folder
    """

    if os.path.exists(dataset_path):
        # Dataset vectorization. From raw .jsonl features to vectorized and labeled data
        thrember.create_vectorized_features(dataset_path, label_type="behavior")
    else:
        print("Dataset folder does not exist")


if __name__ == "__main__":
    DATASET_DIR = "../Dataset"
    download_dataset(dataset_path=DATASET_DIR)
