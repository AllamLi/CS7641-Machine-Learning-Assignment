import matplotlib.pyplot as plt
import numpy as np
import pandas

plot_colors = ('r', 'g', 'b', 'c', 'darkorange', 'black')

def read_data(path_name):
    data = pandas.read_csv(path_name, sep=",")
    size = len(data.values)
    x0 = data.values[0:size, 0]
    x1 = data.values[0:size, 1]
    x2 = data.values[0:size, 2]
    x3 = data.values[0:size, 3]
    return x0, x1, x2, x3

def plot_curves(X, Y, xlabel, ylabel, curve_labels, name, save_name, flag=False):


    plt.figure(figsize=(10, 5))
    plt.title(name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # if (flag):
    #     plt.xticks(x, x)

    for (x, y, label, c) in zip(X, Y, curve_labels, plot_colors):
        plt.plot(x, y, color=c, label=label, lw=2.0)

    plt.legend(loc='best')
    plt.savefig(save_name)
    plt.show()
    return plt

if __name__ == '__main__':
    # it0, ev0, steps0, time0 = read_data("exp_compare_vi.csv")
    # it1, ev1, steps1, time1 = read_data("exp_compare_pi.csv")
    # plot_curves((it0, it1), (time0, time1), 'Grid World Size n', 'Time(ms)', ('Value Iteration', 'Policy Iteration'),
    #             'Grid World Domain Time Complexity: VI vs PI ', '../figures/compare_time.png')
    # plot_curves((it0, it1), (steps0, steps1), 'Grid World Size n', 'Converged Iterations', ('Value Iteration', 'Policy Iteration'),
    #             'Grid World Domain Converged Iterations: VI vs PI ', '../figures/compare_iterations.png')

    # it, ev, steps, time = read_data("MDP1_qlearning.csv")
    #
    # arr = np.zeros(len(it))
    # arr[:] = 88;
    #
    # plot_curves((it, it), (ev, arr), 'Q-Learning Iterations', 'Best Value', ('Q-Learning', 'VI and PI'),
    #             'Easy Grid World Best Reward by Q-Learing', '../figures/mdp1_qlearning.png')
    #

    it, ev, steps, time = read_data("MDP2_qlearning.csv")
    arr1 = np.zeros(len(it))
    arr2 = np.zeros(len(it))

    arr1[:] = -101;
    arr2[:] = -118

    plot_curves((it,it,it), (ev,arr1, arr2), 'Q-Learning Iterations', 'Best Value', ('Q-Learning', 'VI', 'PI'),
                'Complex Grid World Best Reward by Q-Learing', '../figures/mdp2_qlearning.png')