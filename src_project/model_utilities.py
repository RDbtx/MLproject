from sklearn.base import ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from performances import *
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


def model_train(model, labels_names: list, x: np.ndarray, y: np.ndarray):
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

    print("\n----MODEL TRAINING STARTED----")
    print("Training model...")
    train_time = time.time()
    model.fit(x, y)
    print(f"Training time: {(time.time() - train_time):.1f} s")
    print("----TRAINING COMPLETED----\n\n")

    prediction_time = time.time()
    print("\ntraining predictions...")
    train_predictions = model.predict(x)
    print(f"Training predict time: {(time.time() - prediction_time):.1f} s")
    print("predictions computed.\n")

    if type(model) is KNeighborsClassifier:
        accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report, cm = model_performances_multiclass(
            labels_names, y,
            train_predictions, "TRAINING",
            "KNN")
        model_performances_report_generation(accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report,
                                             cm, "TRAINING",
                                             "KNN")
    if type(model) is MultiOutputClassifier:
        accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report, cm = model_performances_multiclass(
            labels_names, y,
            train_predictions, "TRAINING",
            "RF")
        model_performances_report_generation(accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report,
                                             cm, "TRAINING",
                                             "RF")
    elif type(model) is OneVsRestClassifier:
        accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report, cm = model_performances_multiclass(
            labels_names, y,
            train_predictions, "TRAINING",
            "GRAD_BOOSTING")
        model_performances_report_generation(accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report,
                                             cm, "TRAINING",
                                             "GRAD_BOOSTING")

    return model, train_predictions


def model_test(model, labels_names: list, x: np.ndarray, y: np.ndarray):
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
    print(f"Test time: {(time.time() - test_time):.1f} s")
    print("----TESTING COMPLETED----\n\n")

    if type(model) is KNeighborsClassifier:
        accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report, cm = model_performances_multiclass(
            labels_names, y,
            test_predictions, "TESTING",
            "KNN")
        model_performances_report_generation(accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report,
                                             cm, "TESTING",
                                             "KNN")
    elif type(model) is MultiOutputClassifier:
        accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report, cm = model_performances_multiclass(
            labels_names, y, test_predictions,
            "TESTING", "RF")
        model_performances_report_generation(accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report,
                                             cm, "TESTING",
                                             "RF")
    elif type(model) is OneVsRestClassifier:
        accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report, cm = model_performances_multiclass(
            labels_names, y, test_predictions,
            "TESTING", "GRAD_BOOSTING")
        model_performances_report_generation(accuracy, precision, recall, f1_macro, f1_micro, hamm_loss, class_report,
                                             cm, "TESTING",
                                             "GRAD_BOOSTING")

    return test_predictions
