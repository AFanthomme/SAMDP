import option_iteration
from gridworld import GridWorld
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import time

def study_transition_probabilities(n_repeats=2, n_points=15):
    seed = np.random.randint(2**12)
    print('Results obtained with seed = {}'.format(seed))
    np.random.seed(seed)

    terminal_positions = [(8, 8), (0, 4), (8, 0)]
    rewards_range = np.linspace(0.3, 3., n_points)
    # rewards_range = [1.8]
    horizon = 25
    transitions = np.zeros((len(rewards_range), 3, 3))
    exits = np.zeros((len(rewards_range), 3))

    var_transitions = np.zeros((len(rewards_range), 3, 3))
    var_exits = np.zeros((len(rewards_range), 3))

    for idx, rw in enumerate(tqdm.tqdm(rewards_range, leave=False)):
        tmp_transitions = np.zeros((n_repeats, 3, 3))
        tmp_exits = np.zeros((n_repeats, 3))

        for repeat in range(n_repeats):
            symetric_grid = [
                ['', '', '', '', 1., '', '', '', ''],
                ['', '', '', '', '', '', '', '', ''],
                ['x', '', 'x', 'x', 'x', 'x', 'x', '', 'x'],
                ['', '', '', 'x', 'x', 'x', '', '', ''],
                ['', '', '', 'x', 'x', 'x', '', '', ''],
                ['x', '', 'x', 'x', 'x', 'x', 'x', '', 'x'],
                ['', '', '', '', '', '', '', '', ''],
                ['', '', '', 'x', 'x', 'x', '', '', ''],
                [rw, '', '', 'x', 'x', 'x', '', '', rw],
            ]

            example = option_iteration.trainer(symetric_grid, dump_name=None, horizon=horizon, iovi_iters=4000,
                        option_updates=15, epsilon=0.2, monitor=None, noise=0., terminal_positions=terminal_positions)

            try:
                plop = np.loadtxt('sym_grid_clusters.txt')
                plop2 = np.loadtxt('sym_grid_centers.txt')
                example.clustering = plop
                example.cluster_coord = plop2
                # raise FileNotFoundError
            except FileNotFoundError:
                example.aggregate_states(K=3, w=3, nb_it=25000)
                example.show_clustering()
                np.savetxt('sym_grid_clusters.txt', example.clustering)
                np.savetxt('sym_grid_centers.txt', example.cluster_coord)


            example.infer_transitions(it_nb=3000)
            tmp_transitions[repeat] = example.corrected_transitions
            tmp_exits[repeat] = example.exit_fractions

        transitions[idx] = np.mean(tmp_transitions, axis=0)
        var_transitions[idx] = np.std(tmp_transitions, axis=0)
        exits[idx] = np.mean(tmp_exits, axis=0)
        var_exits[idx] = np.std(tmp_exits, axis=0)

    fig = plt.figure()
    ax = plt.subplot()
    colors = ['red', 'firebrick', 'green', 'forestgreen', 'blue', 'royalblue', 'yellow', 'cyan', 'magenta', 'white', 'black']
    relevant_transitions = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] # Handcoded for the clustering we retained
    for idx, transition in enumerate(relevant_transitions):
        ax.errorbar(rewards_range, transitions[:, transition[0], transition[1]], yerr=var_transitions[:, transition[0],
            transition[1]], c=colors[idx], label="Transition {} to {}".format(transition[0] + 1, transition[1] + 1))
    plt.ylim([-0.05, 1.05])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='upper_center', bbox_to_anchor=(1, 0.75))
    plt.xlabel('Value of the reward on the lower two exits')
    plt.ylabel('Transition probability')
    plt.savefig('sym_grid_transitions')

    plt.figure()
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'black']
    for cluster in range(3):
        plt.errorbar(rewards_range, exits[:, cluster], yerr=var_exits[:, cluster],
                     c=colors[cluster], label="Exits from {}".format(cluster + 1))
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Value of the reward on the lower two exits')
    plt.ylabel('Exit probability')
    plt.legend()
    plt.savefig('sym_grid_exits')

    plt.show()


def show_clustering():
    rw=0
    terminal_positions = [(8, 8), (0, 4), (8, 0)]
    horizon =1

    symetric_grid = [
        ['', '', '', '', 1., '', '', '', ''],
        ['', '', '', '', '', '', '', '', ''],
        ['x', '', 'x', 'x', 'x', 'x', 'x', '', 'x'],
        ['', '', '', 'x', 'x', 'x', '', '', ''],
        ['', '', '', 'x', 'x', 'x', '', '', ''],
        ['x', '', 'x', 'x', 'x', 'x', 'x', '', 'x'],
        ['', '', '', '', '', '', '', '', ''],
        ['', '', '', 'x', 'x', 'x', '', '', ''],
        [rw, '', '', 'x', 'x', 'x', '', '', rw],
    ]

    example = option_iteration.trainer(symetric_grid, dump_name=None, horizon=horizon, iovi_iters=4,
                                           option_updates=1, epsilon=0.2, monitor=None, noise=0.,
                                           terminal_positions=terminal_positions)
    plop = np.loadtxt('sym_grid_clusters.txt', dtype=int)
    plop2 = np.loadtxt('sym_grid_centers.txt')
    example.clustering = plop
    example.cluster_coord = plop2
    example.show_clustering()
    time.sleep(100)

if __name__ == '__main__':
    study_transition_probabilities(n_repeats=100, n_points=20)
    # show_clustering()
