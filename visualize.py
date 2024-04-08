import pandas as pd

import matplotlib.pyplot as plt

import classifier
import hdf5_maker


def plot_person_data(person_name: str, data_jump: pd.DataFrame, data_walk: pd.DataFrame):

    # Create a layout for a large 3d plot on the left and two 2d plots on the right
    fig = plt.figure()
    fig.suptitle(f"{person_name} Data")
    gs = fig.add_gridspec(2, 2, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(gs[:, 0], projection='3d')

    # plot the 3d jumping and walking data on top of each other
    jumping, = ax.plot(
        data_jump['Linear Acceleration x (m/s^2)'],
        data_jump['Linear Acceleration y (m/s^2)'],
        data_jump['Linear Acceleration z (m/s^2)'],
        color='r'
    )
    jumping.set_label("Jumping")
    walking, = ax.plot(
        data_walk['Linear Acceleration x (m/s^2)'],
        data_walk['Linear Acceleration y (m/s^2)'],
        data_walk['Linear Acceleration z (m/s^2)'], color='g'
    )
    walking.set_label("Walking")

    ax.legend()
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Directional acceleration')

    # plot the jumping data
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data_jump["Time (s)"], data_jump["Absolute acceleration (m/s^2)"], color='r')
    ax2.set_ylim([-5, 200])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Abs. accel. (m/s^2)')
    ax2.set_title('Jumping abs. acceleration')

    # plot the walking data
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(data_walk["Time (s)"], data_walk["Absolute acceleration (m/s^2)"], color='g')
    ax3.set_ylim([-5, 200])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Abs. accel. (m/s^2)')
    ax3.set_title('Walking abs. acceleration')

    # Show plot
    plt.show()


def visualize(hdf5_filename):
    # get original data
    person_data = hdf5_maker.read_hdf5_original_datasets(hdf5_filename)

    # plot acceleration vs time
    for name, df_list in person_data.items():
        plot_person_data(name, df_list[0], df_list[1])

    # plot preprocessed data
    for name, df_list in person_data.items():
        preprocessed_data = classifier.preprocess(df_list)
        plot_person_data(name, preprocessed_data[0], preprocessed_data[1])

    # get all the samples
    train, test = hdf5_maker.read_hdf5_train_test(hdf5_filename)
    train += test
    # get the features
    features_df = classifier.extract_features(train)

    # plot all features
    plot_features(features_df)


def plot_features(features_df):
    # create one figure that contains a histogram of the X Y Z and A (absolute) acceleration
    # for each feature
    feature_names = [
        "min",
        "max",
        "range",
        "mean",
        "median",
        'sum',
        "std",
        "variance",
        "skew",
        "interquartile",
        "kurtosis",
        "correlation",
    ]
    feature_axes = ["X", "Y", "Z", "A"]

    # this will create a window per feature figure
    for feature_name in feature_names:
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for i, axis in enumerate(axes):
            features_df[f"{feature_name}{feature_axes[i]}"].plot.hist(bins=15, alpha=0.5, ax=axis)
            axis.set_title(f"{feature_name}{feature_axes[i]}")
        plt.tight_layout()
        plt.show()


def main():
    visualize("data.h5")


if __name__ == "__main__":
    main()
