import os

import pandas as pd

import matplotlib.pyplot as plt
import hdf5_maker
import h5py

def plot_data(df: pd.DataFrame):
    data_jump = df.loc[(df["category"] == 0)]
    data_walk = df.loc[(df["category"] == 1)]

    # Create a 3D plot
    fig = plt.figure()
    gs = fig.add_gridspec(2,2, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(gs[:, 0], projection='3d')

    jumping, = ax.plot(data_jump['Linear Acceleration x (m/s^2)'], data_jump['Linear Acceleration y (m/s^2)'], data_jump['Linear Acceleration z (m/s^2)'], color='r')
    jumping.set_label("Jumping")
    walking, = ax.plot(data_walk['Linear Acceleration x (m/s^2)'], data_walk['Linear Acceleration y (m/s^2)'], data_walk['Linear Acceleration z (m/s^2)'], color='g')
    walking.set_label("Walking")

    ax.legend()
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Directional acceleration')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data_jump["Time (s)"], data_jump["Absolute acceleration (m/s^2)"], color='r')
    ax2.set_ylim([-5, 100])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Abs. accel. (m/s^2)')
    ax2.set_title('Jumping abs. acceleration')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(data_walk["Time (s)"], data_walk["Absolute acceleration (m/s^2)"], color='g')
    ax3.set_ylim([-5, 100])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Abs. accel. (m/s^2)')
    ax3.set_title('Walking abs. acceleration')

    # Show plot
    plt.show()


def preprocess(hdf5_filename) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    train = []
    test = []

    with h5py.File("data.h5","r") as hdf:
        train_group = hdf.get("dataset/Train")
        for key in train_group.keys():
            df = pd.DataFrame(train_group[key][:])
            train.append(df)

        test_group = hdf.get("dataset/Test")
        for key in test_group.keys():
            df = pd.DataFrame(test_group[key][:])
            test.append(df)

    window_size = 5
    samples: list[pd.DataFrame] = train + test
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
    return train, test


def visualize(hdf5_filename):
    with h5py.File(hdf5_filename,"r") as hdf:
        person_names = os.listdir("data")

        for person_name in person_names:
            df = pd.DataFrame(hdf[person_name][:])
            plot_data(df)

if __name__ == "__main__":
    hdf5_filename = "data.h5"
    hdf5_maker.create_hdf5("data",hdf5_filename)
    visualize(hdf5_filename)
    preprocess(hdf5_filename)

