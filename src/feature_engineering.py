def combine_features(load_data, frequency_data):

    combined = load_data.join(
        frequency_data,
        how="inner"
    )

    return combined


def remove_label_column(data):

    if "label" in data.columns:
        data = data.drop(
            columns=["label"]
        )

    return data
