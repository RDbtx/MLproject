from model_utilities import *

# =====================================
# --- Model declaration ---
# =====================================

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

knn = KNeighborsClassifier(
    n_neighbors=6,
    weights='distance',
    algorithm='brute',
    metric='euclidean',
    n_jobs=-1
)

# =====================================
# --- Main Execution ---
# =====================================

if __name__ == "__main__":
    DATASET_DIR = "../Dataset"
    EXTRACTED_DATA_DIR = "../Extracted"
    RESULTS_DIR = "../Results"

    x_train, y_train, x_test, y_test, x_challenge, y_challenge = feature_loading(EXTRACTED_DATA_DIR,
                                                                                 ["x_train.npy", "y_train.npy",
                                                                                  "x_test.npy", "y_test.npy"])

    print("\n----DATASET ANALYSIS----")
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    # modify the third variable of the subset_generation function if you want to use a smaller dataset
    train_labels, x_train, y_train = subset_generation(x_train, y_train, len(x_train), "TRAINING")
    test_labels, x_test, y_test = subset_generation(x_test, y_test, len(x_test), "TESTING")

    print("\n----DATASET ANALYSIS----")
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    # this section is used to reshape the test set since seems like that the test set has additional labels
    # that are not present in the training set.
    train_labels, test_labels, x_train, y_train, x_test, y_test = shape_fixer(train_labels, test_labels, x_train,
                                                                              y_train, x_test, y_test)

    subset_analysis(x_train, y_train, "TRAINING", train_labels)
    subset_analysis(x_test, y_test, "TESTING", test_labels)

    # model training
    trained_rf, train_predictions = model_train(knn, train_labels, x_train, y_train)
    save_model(trained_rf, "KNN")

    test_predictions = model_test(knn, test_labels, x_test, y_test)
