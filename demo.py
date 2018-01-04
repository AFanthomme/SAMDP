'''
This script designed to run some simple experiments using the tools from the src module
'''
'''
This script designed to run some simple experiments using the tools from the src module
'''
import src.option_iteration
from src.gridworld import GridWorld

if __name__ == '__main__':

    example = src.option_iteration.test_gridmaker()

    example.aggregate_states(K=5,w=3)
    example.show_clustering()
