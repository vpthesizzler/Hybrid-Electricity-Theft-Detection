from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def train_models(X_train, y_train):

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(64,32),
        max_iter=800,
        random_state=42
    )


    rf.fit(
        X_train,
        y_train
    )

    mlp.fit(
        X_train,
        y_train
    )


    return rf, mlp



def hybrid_probability(rf, mlp, X_test):

    rf_prob = rf.predict_proba(
        X_test
    )[:,1]


    mlp_prob = mlp.predict_proba(
        X_test
    )[:,1]


    hybrid_prob = (
        0.6 * rf_prob +
        0.4 * mlp_prob
    )


    return hybrid_prob
