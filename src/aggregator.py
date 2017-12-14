##################################################
# Antoine Goblet

from src.gridworld import GridWorld1
import src.gridrender as gui
import numpy as np
import time, pdb
import matplotlib.pyplot as plt

"""plt.rcParams.update({'text.usetex' : True,\
                     'font.size' : 17,\
                     'font.weight' : 'bold'})"""

##################################################
# state agregation

gamma = 0.95
delta = 0.01
Tmax = 7 #int(- np.log( delta ) / (1 - gamma))


def collect_trajectory(env, policy=None, horizon=50):
    # Collect a trajectory in an environment
    # A function state -> state (the policy) can be given, otherwise actions chosen at random.
    traj = np.zeros(horizon, dtype=int)
    state = env.reset()

    for t in range(int(horizon)):
        traj[t] = state
        if not policy:
            action = np.random.choice(env.state_actions[state])
        else:
            action = policy(state)
        nexts, reward, term = env.step(state, action)
        state = nexts

        if term:
            traj[t:] = state
            break
    return traj

def aggregate_states(env, K = 5, w=3):
    # K : number of clusters
    # w : size of the sliding window

    ##### random initialization ######

    N = env.n_states
    C_coord = [ env.state2coord[state] for state in np.random.choice(range(N),size=K,replace=False) ]
    print("Initial clusters : {}".format(C_coord))
    state_cluster = np.zeros(N, dtype=int)

    for state in range(N):
        coord = np.array(env.state2coord[state])
        state_cluster[state] = np.argmin([np.linalg.norm(coord - C_coord[k]) for k in range(K)])

    print("Initial clustering : {}".format(state_cluster))

    pol = None

    nb_it = 100
    for it in range(nb_it):
        if it % 20 == 0:
            print(C_coord)
            print(state_cluster)

        # We update the clustering after every trajectory

        # simulate a trajectory
        traj = collect_trajectory(env, policy=pol, horizon=Tmax)

        # update C_coord
        for k in range(K):
            states_k = np.array(env.state2coord)[state_cluster==k]
            C_coord[k] = states_k.sum(axis=0) / float(len(states_k))

        # update clustering
        for state in range(N):
            if state in traj:
                state_idx = np.where(traj==state)[0][0]
                ext_state = traj[int(max(state_idx-w, 0)):int(min(state_idx+w+1, N))]
                ext_state_coord = np.array([env.state2coord[i] for i in ext_state])
                state_cluster[state] = np.argmin([np.linalg.norm(ext_state_coord-C_coord[k]) for k in range(K)])


    print("Final clusters : {}".format(C_coord))
    print("Final clustering : {}".format(state_cluster))

    return C_coord, state_cluster


if __name__ == '__main__':
    env = GridWorld1
    C_coord, state_cluster = aggregate_states(env)