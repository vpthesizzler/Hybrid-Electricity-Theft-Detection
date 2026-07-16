import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)



def calculate_metrics(y_test, prediction, probability):

    return {
        "Accuracy":
        accuracy_score(y_test, prediction),

        "Recall":
        recall_score(y_test, prediction),

        "F1-Score":
        f1_score(y_test, prediction),

        "ROC-AUC":
        roc_auc_score(y_test, probability)
    }



def optimise_threshold(y_test, probability):

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probability
    )

    gmeans = tpr * (1-fpr)

    index = gmeans.argmax()

    return thresholds[index]



def save_confusion_matrix(y_test, prediction):

    cm = confusion_matrix(
        y_test,
        prediction
    )

    plt.figure(
        figsize=(6,5)
    )

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix - Hybrid Model"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i,j]),
                ha="center",
                va="center"
            )


    plt.savefig(
        "results/confusion_matrix_hybrid.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.close()
