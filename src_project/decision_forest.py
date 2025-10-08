from performances import *
from setup_dataset import *
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# --- Main testing function ---
# =========================================================

def model_test(x, y, subset=100000,):
    results_dir = "../Results/decision_forest/"
    os.makedirs(results_dir, exist_ok=True)

    subset = min(subset, len(x))
    idx = np.random.choice(len(x), subset, replace=False)
    x_small = x[idx]
    y_small = y[idx]

    #saving labels
    np.save(results_dir + "x_small.npy", x_small)
    np.save(results_dir + "y_small.npy", y_small)

    subset_analysis(x_small, y_small)

    rf = RandomForestClassifier(
        n_estimators=600,  # good balance for 1M samples
        max_depth=14,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )

    print("\n----MODEL TRAINING STARTED----")
    print("Training model...")
    t0 = time.time()
    rf.fit(x_small, y_small)
    print(f"Training time: {time.time() - t0:.1f} s")
    print("----TRAINING COMPLETED----\n\n")
    # --- Predict probabilities ---
    proba_output = rf.predict_proba(x_small)



    if isinstance(proba_output, list):  # multi-output
        proba = []
        aucs = []
        for k, pk in enumerate(proba_output):
            classes_k = rf.classes_[k]
            pos_idx = list(classes_k).index(1) if 1 in classes_k else (1 if pk.shape[1] > 1 else 0)
            proba.append(pk[:, pos_idx])
            if len(np.unique(y_small[:, k])) == 2:
                aucs.append(roc_auc_score(y_small[:, k], pk[:, pos_idx]))
        proba = np.column_stack(proba)
        if aucs:
            print(f"Mean ROC-AUC across {len(aucs)} labels: {np.mean(aucs):.4f}")
    else:
        classes = rf.classes_
        pos_idx = list(classes).index(1) if 1 in classes else 1
        proba = proba_output[:, pos_idx]
        if len(np.unique(y_small)) == 2:
            auc = roc_auc_score(y_small, proba)
            print(f"AUC: {auc:.4f}")

    # --- Plots ---

    # saves probabilities
    np.save(results_dir + "proba.npy", proba)

    print("\n\n📊 Plotting ROC curve and confusion matrix...")
    plot_roc(y_small, proba)
    plot_conf_matrix(y_small, proba, threshold=0.1)

    evaluate_model(y_small, proba, threshold=0.1)
    return rf, proba


if __name__ == "__main__":
    DATASET_DIR = "../Dataset"
    EXTRACTED_DATA_DIR = "../Extracted"


    x_train, y_train, x_test, y_test, x_challenge, y_challenge = feature_loading(EXTRACTED_DATA_DIR, ["x_train.npy", "y_train.npy"])

    print("----DATASET ANALYSIS----")
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)
    print("x_challenge shape: ", x_challenge.shape)
    print("y_challenge shape: ", y_challenge.shape)


    rf_test, return_proba_test = model_test(x_train, y_train, 50000)
