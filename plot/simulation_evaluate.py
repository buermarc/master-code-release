import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import pandas
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--animate', action="store_true")
    args = parser.parse_args()

    joints = pandas.read_csv("../data/out.csv")
    grouped = joints.values.reshape((len(joints), 32, 3))

    len_samples = len(joints)

    noisy_steps = np.load("../data/noisy_steps.npy")
    steps = np.load("../data/steps.npy")

    grouped = np.load("../data/filtered.npy")
    filtered = grouped
    # grouped = np.load("../simulations/E3/steps.npy")
    len_samples = grouped.shape[0]

    # Attaching 3D axis to the figure

    if not args.animate:
        plt.rcParams["figure.figsize"] = (12,12)
        for i in range(grouped.shape[1]):
            fig, axis = plt.subplots(2, 3)

            axis[0, 0].plot(steps[:, i, 0], label="Steps")
            axis[0, 0].plot(noisy_steps[:, i, 0], label="Noisy Steps")
            axis[0, 0].plot(filtered[:, i, 0], label="Filtered")
            axis[0, 0].set_title(f"Joint {i} - x", fontsize=8)
            axis[0, 0].legend()

            axis[0, 1].plot(steps[:, i, 1], label="Steps")
            axis[0, 1].plot(noisy_steps[:, i, 1], label="Noisy Steps")
            axis[0, 1].plot(filtered[:, i, 1], label="Filtered")
            axis[0, 1].set_title(f"Joint {i} - y", fontsize=8)
            axis[0, 1].legend()

            axis[0, 2].plot(steps[:, i, 2], label="Steps")
            axis[0, 2].plot(noisy_steps[:, i, 2], label="Noisy Steps")
            axis[0, 2].plot(filtered[:, i, 2], label="Filtered")
            axis[0, 2].set_title(f"Joint {i} - z", fontsize=8)
            axis[0, 2].legend()

            axis[1, 0].plot(steps[:, i, 0] - noisy_steps[:, i, 0], label="Diff Simulation and Noise")
            axis[1, 0].plot(steps[:, i, 0] - filtered[:, i, 0], label="Diff Simulation and Filtered")
            axis[1, 0].set_title(f"Joint {i} - x", fontsize=8)
            axis[1, 0].legend()

            axis[1, 1].plot(steps[:, i, 1] - noisy_steps[:, i, 1], label="Diff Simulation and Noise")
            axis[1, 1].plot(steps[:, i, 1] - filtered[:, i, 1], label="Diff Simulation and Filtered")
            axis[1, 1].set_title(f"Joint {i} - y", fontsize=8)
            axis[1, 1].legend()

            axis[1, 2].plot(steps[:, i, 2] - noisy_steps[:, i, 2], label="Diff Simulation and Noise")
            axis[1, 2].plot(steps[:, i, 2] - filtered[:, i, 2], label="Diff Simulation and Filtered")
            axis[1, 2].set_title(f"Joint {i} - z", fontsize=8)
            axis[1, 2].legend()

            plt.savefig(f"results/simualtion-{i}.pdf", )
            plt.cla()


    x = np.arange(steps.shape[0])

    filtered_diff_x = steps[:, :, 0] - filtered[:, :, 0]
    filtered_diff_y = steps[:, :, 1] - filtered[:, :, 1]
    filtered_diff_z = steps[:, :, 2] - filtered[:, :, 2]

    noisy_diff_x = steps[:, :, 0] - noisy_steps[:, :, 0]
    noisy_diff_y = steps[:, :, 1] - noisy_steps[:, :, 1]
    noisy_diff_z = steps[:, :, 2] - noisy_steps[:, :, 2]

    error_filtered = np.linalg.norm(steps - filtered, axis=2)
    rms_filtered = np.sqrt(np.mean(np.power(error_filtered, 2), axis=0))

    error_noisy = np.linalg.norm(steps - noisy_steps, axis=2)
    rms_noisy = np.sqrt(np.mean(np.power(error_noisy, 2), axis=0))

    print(f"rms_filtered: {rms_filtered}")
    print(f"rms_noisy: {rms_noisy}")
    print(f"diff rms: {(rms_noisy - rms_filtered)}")

    if not args.animate:
        fig, axis = plt.subplots(1, 3)

        axis[0].errorbar(x, filtered_diff_x.mean(axis=1), yerr=filtered_diff_x.std(axis=1))
        axis[0].errorbar(x, noisy_diff_x.mean(axis=1), yerr=noisy_diff_x.std(axis=1))

        axis[1].errorbar(x, filtered_diff_y.mean(axis=1), yerr=filtered_diff_y.std(axis=1))
        axis[1].errorbar(x, noisy_diff_y.mean(axis=1), yerr=noisy_diff_y.std(axis=1))

        axis[2].errorbar(x, filtered_diff_z.mean(axis=1), yerr=filtered_diff_z.std(axis=1))
        axis[2].errorbar(x, noisy_diff_z.mean(axis=1), yerr=noisy_diff_z.std(axis=1))



        plt.savefig("results/simualtion_combined_with_errorbar.pdf")
        plt.cla()
        plt.close()

    if args.animate:

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        #idx_order = [0, 2, 1]
        idx_order = [0, 1, 2]
        graph, = ax.plot(grouped[0, :, idx_order[0]].flatten(), grouped[0, :, idx_order[1]].flatten(), (-1)*grouped[0, :, idx_order[2]].flatten(), linestyle="", marker="o")

        def update_graph(num):
            graph.set_data (grouped[num, :, idx_order[0]].flatten(), grouped[num, :, idx_order[1]].flatten())
            # graph.set_3d_properties((-1)*grouped[num, :, idx_order[2]].flatten())
            graph.set_3d_properties(grouped[num, :, idx_order[2]].flatten())
            title.set_text('3D Test, time={}'.format(num))
            return title, graph,


        # Setting the axes properties
        # ax.set_xlim3d([-0.75, 0.75])
        ax.set_xlim3d([grouped[:,:,idx_order[0]].min(), grouped[:,:,idx_order[0]].max()])
        ax.set_xlabel('X')

        # ax.set_ylim([1.0, 2.5])
        ax.set_ylim3d([grouped[:,:,idx_order[1]].min(), grouped[:,:,idx_order[1]].max()])
        ax.set_ylabel('Y')

        # ax.set_zlim3d([-1.0, 1.0])
        ax.set_zlim3d([grouped[:,:,idx_order[2]].min(), grouped[:,:,idx_order[2]].max()])
        ax.set_zlabel('Z')

        title = ax.set_title('3D Test')

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, update_graph, len_samples,
                                           interval=33, blit=False)

        # print(grouped[:, :, 0].min())
        plt.show()

if __name__ == "__main__":
    main()
