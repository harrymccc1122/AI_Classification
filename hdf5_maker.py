import math
import os
import random

import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIRECTORY_PATH = "./data"
SAMPLE_LENGTH = 5


def generate_samples(df) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    end_time = df["Time (s)"].iloc[-1]
    samples = []
    # floor to guarantee all samples have about 5 seconds in them
    number_of_samples = math.floor(end_time/SAMPLE_LENGTH)
    for i in range(0, number_of_samples):
        sample_df = df.loc[(df["Time (s)"] >= i*SAMPLE_LENGTH) & (df["Time (s)"] < (i+1)*SAMPLE_LENGTH)]
        sample_df.loc[:, "Time (s)"] -= i*SAMPLE_LENGTH
        samples.append(sample_df)

    random.shuffle(samples)
    ninety_percent_threshold = math.floor(0.9*len(samples))
    # train, then test
    return samples[0:ninety_percent_threshold], samples[ninety_percent_threshold:]


def create_hdf5(data_directory, hdf5_name):
    person_names = os.listdir(data_directory)
    category_csv_files = os.listdir(f"{data_directory}/{person_names[0]}")
    print(f"Indices of the categories = {enumerate(person_names)}")
    print(category_csv_files)

    with h5py.File(hdf5_name, "w") as hdf:
        for person in person_names:
            person_df_list = []

            for category_index, category_csv_file in enumerate(category_csv_files):
                df = pd.read_csv(f"{data_directory}/{person}/{category_csv_file}")
                df.loc[:, "category"] = category_index
                person_df_list.append(df)

            person_df = pd.concat(person_df_list)
            df_dict = person_df.to_records(index=False)
            hdf.create_dataset(person, data=df_dict)

        training_data: list[pd.DataFrame] = []
        testing_data: list[pd.DataFrame] = []
        for person in person_names:
            for category_index, category_csv_file in enumerate(category_csv_files):
                df = pd.read_csv(f"{data_directory}/{person}/{category_csv_file}")
                df.loc[:, "category"] = category_index
                train, test = generate_samples(df)
                training_data += train
                testing_data += test

        dataset_group = hdf.create_group("dataset")

        training_group = dataset_group.create_group("Train")
        for i, df in enumerate(training_data):
            df_dict = df.to_records(index=False)
            training_group.create_dataset(f"sample_{i}", data=df_dict)

        testing_group = dataset_group.create_group("Test")
        for i, df in enumerate(testing_data):
            df_dict = df.to_records(index=False)
            testing_group.create_dataset(f"sample_{i}", data=df_dict)


def main():
    create_hdf5("data", "data.h5")




if __name__ == "__main__":
    main()