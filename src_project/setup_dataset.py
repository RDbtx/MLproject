from utils import *


def file_mover(origin: str, destination: str) -> None:
    """A utility function that iters to all the file contained into
    the origin directory and moves them to the destination directory.
    This function was created to easily move files from one folder to another.
    Without any overhead of classic drag and drop.

    input:
    origin: path to the origin folder
    destination: path to the destination folder
    """
    for elem in os.listdir(origin):
        if "challenge" in elem:
            print(elem)
            shutil.move(origin + "/" + elem, destination + "/" + elem)


def download_dataset(dataset_path: str) -> None:
    """Download the dataset and extract it into the Dataset folder.
    Once the dataset .jsonl file are all unzipped, the feature extraction
    function is called to transform raw data into vectorized features.

    input:
    dataset_path: path to the dataset folder
    """
    os.makedirs(dataset_path, exist_ok=True)
    thrember.download_dataset(dataset_path, file_type="Win64")
    feature_extraction(dataset_path)


def read_dataset(dataset_path: str):
    """This function reads the .dat vectorized transforms it into numpy ndarrays.
    input:
    dataset_path: Path to the .dat vectorized dataset.

    output:
    dataset: Numpy ndarray with shape (n_samples, n_features)
    """
    x_train, y_train = thrember.read_vectorized_features(dataset_path, subset="train")
    x_test, y_test = thrember.read_vectorized_features(dataset_path, subset="test")
    x_challenge, y_challenge = thrember.read_vectorized_features(dataset_path, subset="challenge")
    return x_train, y_train, x_test, y_test, x_challenge, y_challenge


def feature_memorization(dataset_path: str, extracted_data_dir: str) -> None:
    """
    This function extracts the features extracted from the dataset as numpy ndarrays. Then saves
    them as .npy files for future processing, saving precious extraction time since
    thrember.read_vectorized_features takes quite some time.

    input:
    dataset_path: Path to the dataset folder
    extracted_data_dir: Path where to the store the data

    """
    os.makedirs(extracted_data_dir, exist_ok=True)

    t_loading = time.time()
    print("Loading dataset...")
    x_train, y_train, x_test, y_test, x_challenge, y_challenge = read_dataset(dataset_path)
    np.save(extracted_data_dir + "/x_train.npy", x_train)
    np.save(extracted_data_dir + "/y_train.npy", y_train)
    np.save(extracted_data_dir + "/x_test.npy", x_test)
    np.save(extracted_data_dir + "/y_test.npy", y_test)
    np.save(extracted_data_dir + "/x_challenge.npy", x_challenge)
    np.save(extracted_data_dir + "/y_challenge.npy", y_challenge)
    print("Dataset loaded under /Extracted folder!")
    print(f"Loading time: {(time.time() - t_loading):.1f} s")


def feature_loading(data_file_folder: str, desired_datasets: list) -> np.ndarray:
    """This function loads inside some variables the features extracted from the dataset that are
    stored in memory under the data_file_folder path as .npy files.

    input:
    data_file_folder: Path to the folder where the .npy files are located
    desired_datasets: List of datasets to load

    output:
    x_train, y_train, x_test, y_test, x_challenge, y_challenge: variables containing the extracted features"""

    print("\nLoading data...")
    x_train = np.ndarray(shape=(0, 0))
    y_train = np.ndarray(shape=(0, 0))
    x_test = np.ndarray(shape=(0, 0))
    y_test = np.ndarray(shape=(0, 0))
    x_challenge = np.ndarray(shape=(0, 0))
    y_challenge = np.ndarray(shape=(0, 0))

    loading_time = time.time()
    for elem in os.listdir(data_file_folder):
        if elem.endswith(".npy") and elem in desired_datasets:
            if "x_challenge" in elem:
                x_challenge = np.load(os.path.join(data_file_folder, elem))
            elif "x_train" in elem:
                x_train = np.load(os.path.join(data_file_folder, elem))
            elif "x_test" in elem:
                x_test = np.load(os.path.join(data_file_folder, elem))
            elif "y_challenge" in elem:
                y_challenge = np.load(os.path.join(data_file_folder, elem))
            elif "y_train" in elem:
                y_train = np.load(os.path.join(data_file_folder, elem))
            elif "y_test" in elem:
                y_test = np.load(os.path.join(data_file_folder, elem))
            else:
                "NO CORRECT FILE HAS BEEN FOUND!"
        else:
            pass
    print(f"Loading time: {(time.time() - loading_time):.1f} s")
    print("Features loaded inside variables!")
    return x_train, y_train, x_test, y_test, x_challenge, y_challenge


def feature_extraction(dataset_path: str) -> None:
    """this function checks whether the dataset folder exists, and then calls the thrember
    create_vectorized_features function in order to extract features into vectorized features.

    input:
    dataset_path: path to the dataset folder
    """
    if os.path.exists(dataset_path):
        thrember.create_vectorized_features(dataset_path, label_type="behavior")
    else:
        print("Dataset folder does not exist")


if __name__ == "__main__":
    DATASET_DIR = "../Dataset"
    EXTRACTED_DATA_DIR = "../Extracted"
    download_dataset(DATASET_DIR)
    feature_memorization(DATASET_DIR, EXTRACTED_DATA_DIR)
