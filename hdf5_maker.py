import math
import os
import random
from datetime import datetime

import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIRECTORY_PATH = "./data"
SAMPLE_LENGTH = 5


def generate_samples(df) -> list[pd.DataFrame]:
    end_time = df["Time (s)"].iloc[-1]
    samples = []
    # floor to guarantee all samples have about 5 seconds in them
    number_of_samples = math.floor(end_time/SAMPLE_LENGTH)
    for i in range(0, number_of_samples):
        sample_df = df.loc[(df["Time (s)"] >= i*SAMPLE_LENGTH) & (df["Time (s)"] < (i+1)*SAMPLE_LENGTH)]
        sample_df.loc[:, "Time (s)"] -= i*SAMPLE_LENGTH
        samples.append(sample_df)

    return samples


def create_hdf5(data_directory, hdf5_name):
    # Returns ['David', 'Cousin', 'Noah']
    person_names = os.listdir(data_directory)
    # Returns ['jump.csv', 'walk.csv']
    category_csv_files = os.listdir(f"{data_directory}/{person_names[0]}")


    print(f"Indices of the categories = {list(enumerate(category_csv_files))}")

    with h5py.File(hdf5_name, "w") as hdf:
        for person in person_names:
            person_group = hdf.create_group(f"/{person}")

            for category_csv_file in category_csv_files:
                df = pd.read_csv(f"{data_directory}/{person}/{category_csv_file}")
                person_group.create_dataset(category_csv_file, data=df.to_records(index=False))

        data: list[pd.DataFrame] = []
        for person_index, person in enumerate(person_names):
            for category_index, category_csv_file in enumerate(category_csv_files):
                df = pd.read_csv(f"{data_directory}/{person}/{category_csv_file}")
                df.loc[:, "category"] = category_index
                df.loc[:, "person"] = person_index
                samples = generate_samples(df)
                data += samples

        # create dataset group
        dataset_group = hdf.create_group("dataset")
        # shuffle data
        random.shuffle(data)
        # split data
        ninety_percent_threshold = math.floor(0.9*len(data))
        training_data = data[:ninety_percent_threshold]
        testing_data = data[ninety_percent_threshold:]

        training_group = dataset_group.create_group("Train")
        for i, df in enumerate(training_data):
            training_group.create_dataset(f"Sample_{i}", data=df.to_records(index=False))

        testing_group = dataset_group.create_group("Test")
        for i, df in enumerate(testing_data):
            testing_group.create_dataset(f"Sample_{i}", data=df.to_records(index=False))


def read_hdf5_train_test(hdf5_filename: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    train = []
    test = []

    with h5py.File(hdf5_filename, "r") as hdf:
        train_group = hdf.get("dataset/Train")
        for key in train_group.keys():
            df = pd.DataFrame(train_group[key][:])
            train.append(df)

        test_group = hdf.get("dataset/Test")
        for key in test_group.keys():
            df = pd.DataFrame(test_group[key][:])
            test.append(df)

    return train, test


def read_hdf5_original_datasets(hdf5_filename: str) -> dict[str, list[pd.DataFrame]]:
    data = {}

    with h5py.File(hdf5_filename,"r") as hdf:
        person_groups = [(name, item) for name, item in hdf.items() if not name == "dataset"]
        for name, group in person_groups:
            datasets = [item for name, item in group.items() if isinstance(item, h5py.Dataset)]
            df_list = [pd.DataFrame(dataset[:]) for dataset in datasets]

            data[str(name)] = df_list

    return data


def main():
    random.seed(datetime.now().timestamp())
    create_hdf5("data", "data.h5")


if __name__ == "__main__":
    main()