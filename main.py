import pandas as pd

import matplotlib.pyplot as plt


def main():
    jump_csv_file_path = 'jump_back_pocket.csv'
    walk_csv_file_path = 'walk_back_pocket.csv'

    data_jump = pd.read_csv(jump_csv_file_path)
    data_walk = pd.read_csv(walk_csv_file_path)

    # Create a 3D plot
    fig = plt.figure()
    gs = fig.add_gridspec(2,2, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(gs[:, 0], projection='3d')

    jumping, = ax.plot(data_jump['x'], data_jump['y'], data_jump['z'], color='r')
    jumping.set_label("Jumping")
    walking, = ax.plot(data_walk['x'], data_walk['y'], data_walk['z'], color='g')
    walking.set_label("Walking")

    ax.legend()
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Directional acceleration')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data_jump['time'], data_jump['abs'], color='r')
    ax2.set_ylim([-5, 100])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Abs. accel. (m/s^2)')
    ax2.set_title('Jumping abs. acceleration')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(data_walk['time'], data_walk['abs'], color='g')
    ax3.set_ylim([-5, 100])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Abs. accel. (m/s^2)')
    ax3.set_title('Walking abs. acceleration')

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()