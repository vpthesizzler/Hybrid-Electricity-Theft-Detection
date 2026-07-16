import pandas as pd

from preprocessing import (
    load_data,
    prepare_dataset,
    split_and_balance
)

from train_model import (
    train_models,
    hybrid_probability
)

from evaluate import (
    calculate_metrics,
    optimise_threshold,
    save_confusion_matrix
)

from utils import (
    create_results_folder,
    save_results
)


def main():

    print("Starting Electricity Theft Detection Pipeline")

    # Create results folder
    create_results_folder()


    # ==========================
    # Load Dataset
    # ==========================

    sgcc, freq = load_data(
        "sgcc_ml_ready.csv",
        "merged_frequency_data.csv"
    )


    # ==========================
    # Prepare Dataset
    # ==========================

    X, y = prepare_dataset(
        sgcc,
        freq
    )

    print(
        "Dataset shape:",
        X.shape
    )


    # ==========================
    # Train/Test Split + SMOTE
    # ==========================

    X_train, X_test, y_train, y_test = split_and_balance(
        X,
        y
    )


    # ==========================
    # Train Models
    # ==========================

    rf, mlp = train_models(
        X_train,
        y_train
    )


    # ==========================
    # Hybrid Prediction
    # ==========================

    probability = hybrid_probability(
        rf,
        mlp,
        X_test
    )


    threshold = optimise_threshold(
        y_test,
        probability
    )


    prediction = (
        probability >= threshold
    ).astype(int)


    print(
        "Optimal Threshold:",
        round(threshold,4)
    )


    # ==========================
    # Evaluation
    # ==========================

    metrics = calculate_metrics(
        y_test,
        prediction,
        probability
    )


    results = pd.DataFrame(
        [
            {
                "Model":
                "Hybrid RF-MLP",

                **metrics
            }
        ]
    )


    print(results)


    save_results(
        results
    )


    save_confusion_matrix(
        y_test,
        prediction
    )


    print(
        "Pipeline completed successfully"
    )


if __name__ == "__main__":

    main()
