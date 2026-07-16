import os


def create_results_folder():

    os.makedirs(
        "results",
        exist_ok=True
    )


def save_results(dataframe):

    dataframe.to_csv(
        "results/model_comparison.csv",
        index=False
    )
