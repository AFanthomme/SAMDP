import src.option_iteration
from src.gridworld import GridWorld
import numpy as np
import pickle

def study_transition_probabilities():
    seed = np.random.randint(2**12)
    print('Results obtained with seed = {}'.format(seed))
    np.random.seed(seed)

    terminal_positions = [(8, 8), (0, 4), (8, 0)]
    rewards_range = [1.]
    clusters = None
    horizon = 25

    for rw in rewards_range:
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

        example = src.option_iteration.trainer(symetric_grid, dump_name=None, horizon=horizon, iovi_iters=1000,
         option_updates=20, epsilon=0., monitor=500, noise=0., terminal_positions=terminal_positions)

        if clusters is None:
            example.aggregate_states(K=5,w=5)

        example.show_clustering()
        example.infer_transitions()

        try:
            example.env.render = True
            for _ in range(100):
                example.generate_trajectory(T=horizon)
        except:
            pass

if __name__ == '__main__':
    study_transition_probabilities()