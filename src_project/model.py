from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier

import lightgbm as lgb
from model_utilities import *

# =====================================
# --- Model declaration ---
# =====================================

lghtbm = lgb.LGBMClassifier(
    objective='binary',  # binary for each label
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

multi_lghtbm = OneVsRestClassifier(lghtbm)

rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features="sqrt",
    bootstrap=True,
    max_samples=0.8,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)

rf_multi = MultiOutputClassifier(rf, n_jobs=1)

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

    # modify the third variable of the set_generation function if you want to use a smaller dataset
    train_labels, x_train, y_train = set_generation(x_train, y_train, len(x_train), "TRAINING")
    test_labels, x_test, y_test = set_generation(x_test, y_test, len(x_test), "TESTING")

    # uncomment this section if u want your training and test set to be a split of the original training set
    #  x_train, x_test, y_train, y_test, test_labels = training_set_split( x_train, y_train,train_labels, test_size=0.33)

    print("\n----DATASET ANALYSIS----")
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    # this section is used to reshape the test set since seems like that the test set has additional labels
    # that are not present in the training set.
    train_labels, test_labels, x_train, y_train, x_test, y_test = shape_fixer(100, train_labels, test_labels, x_train,
                                                                              y_train, x_test, y_test)

    subset_analysis(x_train, y_train, "TRAINING", train_labels)
    subset_analysis(x_test, y_test, "TESTING", test_labels)

    # model training
    trained_model, train_predictions = model_train(multi_lghtbm, train_labels, x_train, y_train)
    # save_model(trained_model, "KNN")

    test_predictions = model_test(multi_lghtbm, test_labels, x_test, y_test)
