import math

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, \
    roc_curve, RocCurveDisplay
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

import hdf5_maker


def preprocess(samples: list[pd.DataFrame]) -> list[pd.DataFrame]:
    window_size = 5
    columns_to_apply_filter_to = ["Linear Acceleration x (m/s^2)","Linear Acceleration y (m/s^2)","Linear Acceleration z (m/s^2)","Absolute acceleration (m/s^2)"]

    for sample in samples:
        for column in columns_to_apply_filter_to:
            sample[column] = sample[column].rolling(window_size).mean()
        sample.dropna(inplace=True)

    return samples


def extract_features(train_samples: list[pd.DataFrame], include_labels=True) -> pd.DataFrame:
    columns_to_apply_filter_to = ["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
                                  "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]

    x = columns_to_apply_filter_to[0]
    y = columns_to_apply_filter_to[1]
    z = columns_to_apply_filter_to[2]
    a = columns_to_apply_filter_to[3]

    columns = {
        "X": x,
        "Y": y,
        "Z": z,
        "A": a
    }

    all_features_dict = {}
    for column_id, column_name in columns.items():
        all_features_dict |= {
            f"min{column_id}": [],
            f"max{column_id}": [],
            f"range{column_id}": [],
            f"mean{column_id}": [],
            f"median{column_id}": [],
            f'sum{column_id}': [],
            f"std{column_id}": [],
            f"variance{column_id}": [],
            f"skew{column_id}": [],
            f"interquartile{column_id}": [],
            f"kurtosis{column_id}": [],
            f"correlation{column_id}": []
        }

    if include_labels:
        all_features_dict |= {
            "category": [],
            "person": [],
        }

    for sample in train_samples:
        feature_dict = {}
        for column_id, column_name in columns.items():
            feature_dict |= {
                f"min{column_id}": sample[column_name].min(),
                f"max{column_id}": sample[column_name].max(),
                f"range{column_id}": sample[column_name].max() - sample[column_name].min(),
                f"mean{column_id}": sample[column_name].mean(),
                f"median{column_id}": sample[column_name].median(),
                f'sum{column_id}': sample[column_name].sum(),
                f"std{column_id}": sample[column_name].std(),
                f"variance{column_id}": sample[column_name].var(),
                f"skew{column_id}": sample[column_name].skew(),
                f"interquartile{column_id}": sample[column_name].quantile(0.75) - sample[column_name].quantile(0.25),
                f"kurtosis{column_id}": sample[column_name].kurtosis(),
                f"correlation{column_id}": sample[column_name].corr(sample[a])
            }

        if include_labels:
            feature_dict |= {
                "category": sample["category"].max(),
                "person": sample["person"].max(),
            }

        for key, value in feature_dict.items():
            all_features_dict[key].append(value)

    return pd.DataFrame(all_features_dict)


def logistic_regression(df: pd.DataFrame) -> Pipeline:
    data = df.iloc[:, :-2]
    labels = df.loc[:, "category"]
    l_reg = LogisticRegression(max_iter=10000)
    clf = make_pipeline(StandardScaler(), l_reg)
    clf.fit(data, labels)

    return clf


def get_classifier(hdf5_filename):
    train_samples, _test_samples = hdf5_maker.read_hdf5_train_test(hdf5_filename)

    filtered_train_samples = preprocess(train_samples)
    train_features = extract_features(filtered_train_samples)

    clf = logistic_regression(train_features)

    return clf


def classify_data(hdf5_filename: str, csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    correct_column_set = {
        "Time (s)",
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)",
        "Absolute acceleration (m/s^2)"
    }

    if set(df.columns) != correct_column_set:
        raise Exception("Incorrect csv format")

    sample_length = 150
    number_of_samples = math.floor(len(df)/sample_length)
    sample_list = [df.iloc[i*sample_length:(i+1)*sample_length] for i in range(0, number_of_samples)]

    filtered_samples = preprocess(sample_list)

    features = extract_features(filtered_samples, include_labels=False)
    print(features)

    clf = get_classifier(hdf5_filename)
    y_prediction = clf.predict(features)
    print(y_prediction)

    # add column
    df["category"] = None

    for index, prediction in enumerate(y_prediction):
        df.loc[index*sample_length:(index+1)*sample_length, "category"] = prediction

    return df


def main():
    hdf5_filename = "data.h5"
    # hdf5_maker.create_hdf5("data", hdf5_filename)
    train_samples, test_samples = hdf5_maker.read_hdf5_train_test(hdf5_filename)

    filtered_train_samples = preprocess(train_samples)
    filtered_test_samples = preprocess(test_samples)

    train_features = extract_features(filtered_train_samples)
    test_features = extract_features(filtered_test_samples)

    clf = logistic_regression(train_features)

    test_data = test_features.iloc[:, :-2]
    test_labels = test_features.loc[:, "category"]
    y_prediction = clf.predict(test_data)
    y_probability = clf.predict_proba(test_data)

    print(f"{accuracy_score(test_labels, y_prediction) * 100:.1f}% of the predictions are correct")
    cm = confusion_matrix(test_labels, y_prediction)
    print(cm)
    print(f"f1 score is {f1_score(test_labels, y_prediction):.4f}")
    print(f"auc score is {roc_auc_score(test_labels, y_probability[:, 1]):.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["jumping", "walking"])
    disp.plot()
    fpr, tpr, _ = roc_curve(test_labels, y_probability[:, 1], pos_label=clf.classes_[1])
    roc = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc.plot()
    plt.show()

    data = train_features.iloc[:, :-2]
    labels = train_features.loc[:, "category"]
    print(data)
    print(labels)
    l_reg = LogisticRegression(max_iter=10000)
    clf = make_pipeline(StandardScaler(), l_reg)
    clf.fit(data, labels)

    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))

    pipeline.fit(data, labels)
    x_train_pca = pipeline.transform(data)

    clf = make_pipeline(LogisticRegression(max_iter=10000))

    clf.fit(x_train_pca, labels)

    y_prediction = clf.predict(x_train_pca)
    y_probability = clf.predict_proba(x_train_pca)

    disp = DecisionBoundaryDisplay.from_estimator(
        clf, x_train_pca, response_method="predict",
        xlabel='X1', ylabel='X2'
    )
    disp.ax_.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=labels)
    plt.show()

    for index, category_probabilities in enumerate(y_probability):
        print(f"Sample {index}: {max(category_probabilities)*100:.1f}% confident, {"correct" if y_prediction[index] == test_labels[index] else "incorrect"}")

    correct = sum([1 if prediction == real else 0 for prediction, real in zip(y_prediction, test_labels)])
    print(f"{correct / len(test_data) * 100:.1f}% of the predictions are correct")


if __name__ == "__main__":
    main()
    # show_ui()





