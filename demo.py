'''
This script designed to run some simple experiments using the tools from the src module

Implementation : Arnaud Fanthomme and Antoine Goblet
'''

import src.option_iteration
from src.gridworld import GridWorld
import time
import numpy as np
import pickle

def pause():
    programPause = input("Press the <ENTER> key to continue...")

if __name__ == '__main__':
    seed = np.random.randint(2**12)
    print('Results obtained with seed = {}'.format(seed))
    np.random.seed(seed)

    horizon = 25
    full_grid = \
        [
            [4., 'x', '', 'x', '', '', 'x', '', '', '', '', 1],
            ['', 'x', '', '', '', '', 'x', '', '', '', '', ''],
            ['', 'x', '', 'x', '', '', 'x', 'x', 'x', 'x', 'x', ''],
            ['', '', '', 'x', '', '', '', '', '', '', 'x', ''],
            ['x', '', 'x', 'x', '', '', '', '', '', '', 'x', ''],
            ['', 'x', '', '', '', 'x', '', -1, 'x', 'x', 'x', ''],
            ['', 'x', '', '', '', 'x', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', '', '', '', '', ''],
            ['', '', 1, '', '', 'x', '', '', '', '', '', ''],
        ]
    
    # example = src.option_iteration.trainer(full_grid, dump_name='full_grid_4', horizon=horizon, iovi_iters=4000,
    #                                        option_updates=30, epsilon=0., monitor=5000, noise=0.)
    example = pickle.load(open('saves/full_grid_4_verygoodboy.pkl', 'rb'))
    example.aggregate_states(K=5,w=3)
    example.show_clustering()
    example.infer_transitions()
    
    try:
        example.env.render = True
        for _ in range(100):
            example.generate_trajectory(T=horizon)
    except:
        pass
