from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from performances import *
import time
from sklearn.ensemble import RandomForestClassifier
from setup_dataset import feature_loading
from utils import subset_analysis, subset_generation, shape_fixer
from performances import *
import time
from sklearn.ensemble import RandomForestClassifier
from setup_dataset import feature_loading
from utils import subset_analysis, subset_generation, shape_fixer
import joblib


# =====================================
# --- Main utility functions ---
# =====================================


def save_model(model: RandomForestClassifier, model_name: str) -> None:
    """This function is used to save the trained model as a joblib file
    inside the /Models directory

    inputs:
    model: trained model
    model_name: name of the model to be saved

    """
    models_dir = "../Models/"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, models_dir + f"{model_name}_malware_classifier.joblib")


def model_train(model, labels_names: list, x: np.ndarray, y: np.ndarray, results_dir: str):
    """This function is used to train the model evaluate the training performances.

    inputs:
    model: the model to be trained
    x: np.ndarray containing the training set with samples and features
    y: np.ndarray containing the training set with samples and labels
    results_dir: directory to save the training results

    outputs:
    model: the trained model
    train_predictions:  np.ndarray containing the predictions generated during training
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n----MODEL TRAINING STARTED----")
    print("Training model...")
    train_time = time.time()
    model.fit(x, y)
    print(f"Training time: {(time.time() - train_time):.1f} s")
    print("----TRAINING COMPLETED----\n\n")

    print("training predictions...")
    train_predictions = model.predict(x)
    print("predictions computed.")

    # np.save(results_dir + "train_prediction.npy", train_predictions)

    if type(model) is KNeighborsClassifier:
        accuracy, precision, recall, class_report, cm = model_performances_multiclass_knn(labels_names, y,
                                                                                          train_predictions, "TESTING")
    elif type(model) is RandomForestClassifier:
        accuracy, precision, recall, class_report, cm = model_performances_multiclass(labels_names, y,
                                                                                      train_predictions, "TESTING")

    model_performances_report_generation(accuracy, precision, recall, class_report, cm, "TESTING", results_dir)

    return model, train_predictions


def model_test(model, labels_names: list, x: np.ndarray, y: np.ndarray, results_dir: str):
    """This function is used to train the model evaluate the training performances.

        inputs:
        model: the model to be trained
        x: np.ndarray containing the training set with samples and features
        y: np.ndarray containing the training set with samples and labels
        results_dir: directory to save the testing results

        outputs:
        test_predictions: np.ndarray containing the predictions generated during testing
        """
    print("\n----MODEL TESTING----")
    print("Testing model...")
    test_time = time.time()
    test_predictions = model.predict(x)
    print(f"Test time: {(time.time() - test_time):.1f} h")
    print("----TESTING COMPLETED----\n\n")

    # np.save(results_dir + "test_prediction.npy", test_predictions)

    if model is KNeighborsClassifier:
        accuracy, precision, recall, class_report, cm = model_performances_multiclass_knn(labels_names, y,
                                                                                          test_predictions, "TESTING")
        model_performances_report_generation(accuracy, precision, recall, class_report, cm, "TESTING", results_dir)
    elif model is RandomForestClassifier:
        accuracy, precision, recall, class_report, cm = model_performances_multiclass(labels_names, y, test_predictions,
                                                                                      "TESTING")
        model_performances_report_generation(accuracy, precision, recall, class_report, cm, "TESTING", results_dir)

    return test_predictions
