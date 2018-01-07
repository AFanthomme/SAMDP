'''
This script designed to run some simple experiments using the tools from the src module

Implementation : Arnaud Fanthomme and Antoine Goblet
'''

import src.option_iteration
from src.gridworld import GridWorld
import time

def pause():
    programPause = input("Press the <ENTER> key to continue...")

if __name__ == '__main__':

    example = src.option_iteration.test_gridmaker()
    example.aggregate_states(K=5,w=3)
    example.show_clustering()
    # pause()

    example.env.render = True
    for _ in range(25):
        example.generate_trajectory()
