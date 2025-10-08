from performances import *
import time
from sklearn.ensemble import RandomForestClassifier
from setup_dataset import feature_loading
from utils import subset_analysis, subset_generation, shape_fixer
import joblib


# =========================================================
# --- Main testing function ---
# =========================================================

def RandomForest_model_generation() -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=14,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )

    return rf


def save_model(model: RandomForestClassifier):
    models_dir = "../Models/"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, models_dir + "RF_malware_classifier.joblib")


def model_train(model: RandomForestClassifier, x: np.ndarray, y: np.ndarray, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    print("\n----MODEL TRAINING STARTED----")
    print("Training model...")
    train_time = time.time()
    rf.fit(x, y)
    print(f"Training time: {time.time() - train_time:.1f} s")
    print("----TRAINING COMPLETED----\n\n")
    # --- Predict probabilities ---
    train_predictions = model.predict(x)

    # saves probabilities
    np.save(results_dir + "train_prediction.npy", train_predictions)

    accuracy, precision, recall, class_report, cm = model_performances_multiclass(y, train_predictions, "TRAINING")
    model_performances_report_generation(accuracy, precision, recall, class_report, cm, "TRAINING", results_dir)

    return model, train_predictions


def model_test(model: RandomForestClassifier, x: np.ndarray, y: np.ndarray, results_dir: str):
    print("\n----MODEL TESTING----")
    print("Testing model...")
    test_time = time.time()
    test_predictions = model.predict(x)
    print(f"Testing time: {time.time() - test_time:.1f} s")
    print("----TESTING COMPLETED----\n\n")

    np.save(results_dir + "test_prediction.npy", test_predictions)

    accuracy, precision, recall, class_report, cm = model_performances_multiclass(y, test_predictions, "TESTING")
    model_performances_report_generation(accuracy, precision, recall, class_report, cm, "TESTING", results_dir)

    return test_predictions


if __name__ == "__main__":
    DATASET_DIR = "../Dataset"
    EXTRACTED_DATA_DIR = "../Extracted"
    RESULTS_DIR = "../Results/decision_forest/"

    x_train, y_train, x_test, y_test, x_challenge, y_challenge = feature_loading(EXTRACTED_DATA_DIR,
                                                                                 ["x_train.npy", "y_train.npy",
                                                                                  "x_test.npy", "y_test.npy"])
    print("\n----DATASET ANALYSIS----")
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    # this section is used to reshape the test set since seems like that the test set has one additional label
    # that is not present in training.
    shape_fixer(y_train, y_test)

    # uncomment this part of the code if you want to use a smaller dataset
    x_train, y_train = subset_generation(x_train, y_train, 500, RESULTS_DIR)
    x_test, y_test = subset_generation(x_test, y_test, 500, RESULTS_DIR)

    # remove unlabeled samples
    x_train, y_train = subset_analysis(x_train, y_train, "TRAINING")

    rf = RandomForest_model_generation()
    rf, train_predictions = model_train(rf, x_train, y_train, RESULTS_DIR)
    save_model(rf)

    # remove unlabeled samples
    x_test, y_test = subset_analysis(x_test, y_test, "TESTING")
    test_predictions = model_test(rf, x_test, y_test, RESULTS_DIR)
