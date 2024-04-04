import os
import time

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

import hdf5_maker
import h5py
import tkinter as tk


def preprocess(samples: list[pd.DataFrame]) -> list[pd.DataFrame]:

    window_size = 5
    columns_to_apply_filter_to = ["Linear Acceleration x (m/s^2)","Linear Acceleration y (m/s^2)","Linear Acceleration z (m/s^2)","Absolute acceleration (m/s^2)"]

    for sample in samples:
        # plt.plot(sample["Time (s)"], sample["Absolute acceleration (m/s^2)"], color='r')
        for column in columns_to_apply_filter_to:
            sample[column] = sample[column].rolling(window_size).mean()
        sample.dropna(inplace=True)
        # plt.plot(sample["Time (s)"], sample["Absolute acceleration (m/s^2)"], color='b')
        # plt.show()

    # print(len(train))
    # print(len(test))
    return samples


def extract_features(train_samples: list[pd.DataFrame]) -> pd.DataFrame:
    columns_to_apply_filter_to = ["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
                                  "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]

    all_features_dict = {
            "minX": [],
            "maxX": [],
            "meanX": [],
            "stdX": [],
            "kurtosisX": [],
            "minY": [],
            "maxY": [],
            "meanY": [],
            "stdY": [],
            "kurtosisY": [],
            "minZ": [],
            "maxZ": [],
            "meanZ": [],
            "stdZ": [],
            "kurtosisZ": [],
            "minA": [],
            "maxA": [],
            "meanA": [],
            "stdA": [],
            "kurtosisA": [],
            "category": [],
            "person": [],
        }

    for sample in train_samples:
        x = columns_to_apply_filter_to[0]
        y = columns_to_apply_filter_to[1]
        z = columns_to_apply_filter_to[2]
        a = columns_to_apply_filter_to[3]

        feature_dict = {
            "minX": sample[x].min(),
            "maxX": sample[x].max(),
            "meanX": sample[x].mean(),
            "stdX": sample[x].std(),
            "kurtosisX": sample[x].kurtosis(),
            "minY": sample[y].min(),
            "maxY": sample[y].max(),
            "meanY": sample[y].mean(),
            "stdY": sample[y].std(),
            "kurtosisY": sample[y].kurtosis(),
            "minZ": sample[z].min(),
            "maxZ": sample[z].max(),
            "meanZ": sample[z].mean(),
            "stdZ": sample[z].std(),
            "kurtosisZ": sample[z].kurtosis(),
            "minA": sample[a].min(),
            "maxA": sample[a].max(),
            "meanA": sample[a].mean(),
            "stdA": sample[a].std(),
            "kurtosisA": sample[a].kurtosis(),
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

def show_ui():
    window = tk.Tk()
    greeting = tk.Label(text="Hello, Tkinter")
    greeting.pack()

def main():
    hdf5_filename = "data.h5"
    hdf5_maker.create_hdf5("data", hdf5_filename)
    # visualize(hdf5_filename)
    train_samples, test_samples = hdf5_maker.read_hdf5_train_test(hdf5_filename)

    filtered_train_samples = preprocess(train_samples)
    filtered_test_samples = preprocess(test_samples)

    train_features = extract_features(filtered_train_samples)
    test_features = extract_features(filtered_test_samples)

    clf = logistic_regression(train_features)

    test_data = test_features.iloc[:, :-2]
    test_labels = test_features.loc[:, "category"]
    y_prediction = clf.predict(test_data)
    y_clf_prob = clf.predict_proba(test_data)

    for index, category_probabilities in enumerate(y_clf_prob):
        print(f"Sample {index}: {max(category_probabilities)*100:.1f}% confident, {"correct" if y_prediction[index] == test_labels[index] else "incorrect"}")

    correct = sum([1 if prediction == real else 0 for prediction, real in zip(y_prediction, test_labels)])
    print(f"{correct / len(test_data) * 100:.1f}% of the predictions are correct")


if __name__ == "__main__":
    main()
    # show_ui()





