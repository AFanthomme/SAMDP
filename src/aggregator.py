##################################################
## GOBLET Antoine

from gridworld import GridWorld1, GridWorld2
import gridrender as gui
import numpy as np
import time, pdb
import matplotlib.pyplot as plt

"""plt.rcParams.update({'text.usetex' : True,\
                     'font.size' : 17,\
                     'font.weight' : 'bold'})"""

env = GridWorld2

##################################################
# state agregation

gamma = 0.95
delta = 0.01
Tmax = - np.log( delta ) / (1 - gamma)

##### random initialization ######
K = 5 # number of clusters
N = env.n_states
w = 3 # moving window
states = np.arange(N)
C_coord = [ env.state_coord[state] for state in np.random.choice(range(N),size=K,replace=False) ]
print("Initial clusters : {}".format(C_coord))
state_cluster = np.zeros(N).astype(int)
for state in range(N):
        coord = np.array(env.state_coord[state])
        state_cluster[state] = np.argmin([np.linalg.norm(coord - C_coord[k]) for k in range(K)])
print("Initial clustering : {}".format(state_cluster))

#pol = [0,0,0,0,3,0,0,0,0,0,3]

nb_it = 100
for it in range(nb_it):
        # simulate and store one trajectory
        traj = []
        state = env.reset()
        for t in range(int(Tmax)):
                traj.append(state)
                if state in (3,6):
                        break
                #action = pol[state]
                action = np.random.choice( env.state_actions[state] )
                nexts, reward, term = env.step(state,action)
                state = nexts

        # update C_coord
        for k in range(K):
                states_k = np.array(env.state_coord)[state_cluster==k]
                C_coord[k] = states_k.sum(axis=0) / float(len(states_k))

        # update clustering
        for state in range(N):
                if state in traj:
                        state_idx = traj.index(state)
                        ext_state = traj[ max(state_idx-w,0) : state_idx+w+1 ]
                        ext_state_coord = np.array([ env.state_coord[i] for i in ext_state ])
                        state_cluster[state] = np.argmin([np.linalg.norm(ext_state_coord-C_coord[k]) for k in range(K)])

print("Final clusters : {}".format(C_coord))
print("Final clustering : {}".format(state_cluster))

env.show_clustering(C_coord,state_cluster)