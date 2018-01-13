'''
Implementation of the Time-Regularized Interrupting options framework
Ref : Mankowitz, D. J.; Mann, T. A.; and Mannor, S. 2014. Time regularized interrupting options. International
      Conference on Machine Learning

Implementation : Arnaud Fanthomme and Antoine Goblet
'''

import numpy as np
from src.gridworld import GridWorld
import matplotlib.pyplot as p
import src.gridrender as gui
from copy import deepcopy
import pickle, pdb
from tkinter import Tk
import tkinter.font as tkfont
import numbers
import threading

def gridmaker(grid, terminal_positions):
    env = GridWorld(grid=grid, gamma=0.95, time_penalty=0.0, noise=0.1)
    terminal_states = [env.coord2state[position[0], position[1]] for position in terminal_positions]
    options_mdp = option_gridworld(env, terminal_states=terminal_states)
    problematic_states = [[state for state in range(env.n_states) if action not in env.state_actions[state] or
                           state in terminal_states] for action in range(4)]

    # Remove from inits states where a certain action cannot be initiated
    for action in range(4):
        options_mdp.options[action]['init'] = np.delete(options_mdp.options[action]['init'], problematic_states[action])

    # Add termination probabilities of 1 for states where a certain action is forbidden
    for action in range(4):
        options_mdp.options[action]['term'][problematic_states[action]] = 1.

    return options_mdp

class option_gridworld:
    def __init__(self, grid=None, env=None, terminal_states=None, terminal_positions=None, noise=0.):
        self.terminal_states = terminal_states

        if grid is None:
            if env is None:
                raise RuntimeError('Please provide either a grid or a Gridworld environment')
            self.env = env
        else:
            self.env = GridWorld(grid=grid, gamma=0.95, time_penalty=0.0, noise=noise)

        self.n_states = self.env.n_states
        # Default options are going straight in one direction, can be started anywhere and never terminate.
        self.n_options = 4
        self.options = [
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 0 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 1 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 2 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 3 * np.ones(self.n_states, dtype=int)}]

        if self.terminal_states is None:
            self.terminal_states = [self.env.coord2state[position[0], position[1]] for position in terminal_positions]

        problematic_states = [[state for state in range(self.env.n_states) if action not in self.env.state_actions[state] or
                               state in self.terminal_states] for action in range(4)]
        for action in range(4):
            self.options[action]['init'] = np.delete(self.options[action]['init'], problematic_states[action])

        for action in range(4):
            self.options[action]['term'][problematic_states[action]] = 1.

    def q_estimation(self, n_epochs=100, T=30, epsilon=0., seed=None):
        env = self.env
        n_actions = 4  # Hardcode this for simplicity as in ex_1
        if seed:
            np.random.seed(seed)
        ep_length = T # This influences the final options quite a lot
        nb_encounters = np.zeros((env.n_states, n_actions))
        q_estimate = np.zeros((env.n_states, n_actions))

        alpha = lambda state, action: 1. / nb_encounters[state, action]

        for epoch in range(n_epochs):
            states, skills, next_states, rewards = self.generate_trajectory(T=ep_length, noise=epsilon, resetting=True)
            idx, skill_reward = 0, 0

            for state, skill, next_state, reward in zip(states, skills, next_states, rewards):
                if idx == 0:
                    previous_option = skill
                    initiating_state = state

                elif idx == len(states) - 1:
                    skill_reward += reward
                    nb_encounters[initiating_state, previous_option] += 1
                    alpha_k = alpha(initiating_state, previous_option)
                    q_estimate[initiating_state, previous_option] = (1. - alpha_k) * q_estimate[
                        initiating_state, previous_option] + alpha_k * \
                        (skill_reward + env.gamma * np.max(q_estimate[state, :]))
                    # I had forgotten that part which make the estimate fo the terminal states non-zero...
                    nb_encounters[state, skill] += 1
                    alpha_k = alpha(state, skill)
                    q_estimate[state, skill] = (1. - alpha_k) * q_estimate[state, skill] + \
                            alpha_k * (skill_reward + env.gamma * np.max(q_estimate[next_state, :]))

                elif previous_option != skill:
                    nb_encounters[initiating_state, previous_option] += 1
                    alpha_k = alpha(initiating_state, previous_option)
                    q_estimate[initiating_state, previous_option] = (1. - alpha_k) * \
                                            q_estimate[initiating_state, previous_option] + alpha_k * \
                                            (skill_reward + env.gamma * np.max(q_estimate[state, :]))
                    previous_option = skill
                    skill_reward = reward
                    initiating_state = state
                else:
                    skill_reward += reward
                idx += 1

        return q_estimate


    def IOVI(self, vi_steps=100, option_updates=100, horizon=30, monitor_performance=None, epsilon=0.):
        """
        Use the IOVI procedure to train skills by interrrupting the initial options.

        :param initial_options:
        :param regularizer:
        :return:
        """

        if monitor_performance:
            performance_record = np.zeros(option_updates)

        initial_options = deepcopy(self.options)
        current_options = self.options
        alpha = np.ones((self.n_options, self.n_states))


        for t in range(option_updates):
            if monitor_performance:
                performance_record[t] = np.mean([np.sum(self.generate_trajectory(T=horizon)[3]) for
                                                 _ in range(monitor_performance)])

            if t%(option_updates//10) == 0:
                print('{}% done'.format((100.*t)//option_updates))
            # Since we do not have access to the real transition probabilities, we use the Q-learning procedure from TD1
            # instead of VI to obtain the new Q function.
            current_Q = self.q_estimation(n_epochs=vi_steps, T=horizon, epsilon=epsilon)

            for idx, option in enumerate(current_options):
                option['term'] = np.maximum(initial_options[idx]['term'],
                                (current_Q[:, idx] < np.max(current_Q, axis=1)).astype(float))

        if monitor_performance:
            p.plot(performance_record)
            p.title('Evolution of the performance as a function of the number of iterations')
            p.show()

        return current_options

    def plot_terminations(self):
        f, axarr = p.subplots(self.n_options, sharex=True)
        names = ['right', 'down', 'left', 'up']
        for option in range(self.n_options):
            axarr[option].bar(range(self.n_states), self.options[option]['term'], label='Option {}'.format(option),
                              align='center')
            axarr[option].set_title('Option {} : always go {}'.format(option, names[option]))
        p.show()

    def render_terminations(self):
        policy_array = [[1. - self.options[option]['term'][state] for option in range(4)]
                         for state in range(self.n_states)]
        gui.render_non_det_policy(self.env, np.array(policy_array, dtype=int))


    def choose_skill(self, state):
        possible_actions = [i for i in range(self.n_options) if state in self.options[i]['init']
                             and self.options[i]['term'][state] < 0.99]
        try:
            return np.random.choice(possible_actions)
        except ValueError:
            raise RuntimeError('No actions possible in state {}'.format(state))

    def generate_trajectory(self, T=30, noise=0., resetting=False):
        env = self.env
        state = env.reset()

        # Depending on resetting, accept or not the trajectories that start on a terminal state

        while state in self.terminal_states:
            if not resetting:
                i, j = env.state2coord[state]
                return np.array([state]), np.array([np.random.randint(4)]), np.array([state]), np.array([env.grid[i][j]])
            else:
                state = env.reset()

        skill_in_use = self.choose_skill(state)
        termination_probs = self.options[skill_in_use]['term']
        states = np.zeros(T, dtype=int)
        rewards = np.zeros(T)
        next_states = np.zeros(T, dtype=int)
        skills = np.zeros(T, dtype=int)

        for t in range(T):
            states[t] = state
            skills[t] = skill_in_use
            nextstate, reward, term = env.step(state, skill_in_use)
            next_states[t] = nextstate
            rewards[t] = reward
            state = nextstate

            if term:
                try:
                    next_states[t+1] = state
                    states[t+1] = state
                    rewards[t+1] = 0
                    skills[t+1] = skill_in_use
                except IndexError:
                    t-=1

                next_states = next_states[:t+2]
                states = states[:t+2]
                rewards = rewards[:t+2]
                skills = skills[:t+2]
                break
            elif np.random.rand() < termination_probs[state] or np.random.rand() < noise:
                # With probability termination_probs[state], interrupt option and choose a new one
                skill_in_use = self.choose_skill(state)
                termination_probs = self.options[skill_in_use]['term']

        return states, skills, next_states, rewards

    def aggregate_states(self, K, w, nb_it=500):
        # K : number of clusters
        # w : size of the sliding window

        ##### random initialization ######

        N = self.env.n_states
        cluster_coord = [ self.env.state2coord[state] for state in np.random.choice(range(N),size=K,replace=False) ]
        clustering = np.zeros(N, dtype=int)
        not_encountered_states = list(range(N))
            
        for state in range(N):
            coord = np.array(self.env.state2coord[state])
            clustering[state] = np.argmin([np.linalg.norm(coord - cluster_coord[k]) for k in range(K)])

        for it in range(nb_it):
            # We update the clustering after every trajectory

            # simulate a trajectory
            traj = self.generate_trajectory(T=100, noise=0.)[0]

            # update cluster_coord
            for k in range(K):
                states_k = np.array(self.env.state2coord)[clustering==k]
                if len(states_k)==0: # then no state is assigned to cluster k
                    continue
                cluster_coord[k] = states_k.sum(axis=0) / float(len(states_k))

            # update clustering

            for state_idx,state in enumerate(traj): # every-visit MC
                ext_state = traj[int(max(state_idx-w,0)):int(min(state_idx+w+1,len(traj)))]
                ext_state_coord = np.array([self.env.state2coord[i] for i in ext_state])
                clustering[state] = np.argmin([np.linalg.norm(ext_state_coord-cluster_coord[k]) for k in range(K)])
                if state in not_encountered_states:
                    not_encountered_states.remove(state)            

        if len(not_encountered_states):
            print("warning : the following states were not encountered during the state aggregation step : {}".format(not_encountered_states))
                   
        self.cluster_coord = cluster_coord
        self.clustering = clustering

    def show_clustering(self,print_state=True):

        if hasattr(self, 'cluster_coord'):
            dim = 70
            rows, cols = len(self.env.grid) + 0.5, max(map(len, self.env.grid))
            if hasattr(self, 'window'):
                del(self.window)

            root = Tk()
            self.window = gui.GUI(root)

            self.window.config(width=cols * (dim + 12), height=rows * (dim + 12))
            my_font = tkfont.Font(family="Arial", size=32, weight="bold")
            for s in range(self.env.n_states):
                r, c = self.env.state2coord[s]
                x, y = 10 + c * (dim + 4), 10 + r * (dim + 4)
                if isinstance(self.env.grid[r][c], numbers.Number):
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='blue', width=2)
                    self.window.create_text(x + dim / 2., y + dim / 2., text="{:.1f}".format(self.env.grid[r][c]),
                                            font=my_font, fill='white')
                else:
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='white', width=2)
            self.window.pack()

            my_font = tkfont.Font(family="Arial", size=32, weight="bold")

            # cluster_color = []
            # r = lambda: np.random.randint(0,255)
            # for cluster_idx in range(len(self.cluster_coord)):
            #     cluster_color.append('#%02X%02X%02X' % (r(),r(),r()))

            # Use fixed color cycle for more reproducibility
            cluster_color = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'black']

            self.polygons = []
            for state, cluster_idx in enumerate(self.clustering):
                r, c = self.env.state2coord[state]
                x, y = 10 + c * (dim + 4), 10 + r * (dim + 4)
                self.polygons.append( self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim],
                                                        outline='black', fill=cluster_color[cluster_idx], width=2) )
                if print_state:
                    self.window.create_text(x + dim / 2., y + dim / 2., text="{}".format(state), font=my_font, fill='white')

            self.ovals = []
            for cluster_idx,cluster in enumerate(self.cluster_coord):
                r1, c1 = 10 + cluster[1] * (dim + 4), 10 + cluster[0] * (dim + 4)
                x1, y1 = r1 + dim / 2., c1 + dim / 2.
                self.ovals.append( self.window.create_oval(x1 - dim / 5., y1 - dim / 5., x1 + dim / 5., y1 + dim / 5.,
                                                           fill=cluster_color[cluster_idx]) )

            for oval in self.ovals:
                if hasattr(self, str(oval)):
                    self.window.delete(oval)
            for polygon in self.polygons:
                if hasattr(self, str(polygon)):
                    self.window.delete(polygon)

            self.window.update()
        else:
            print("Error : first run method 'aggregate_states' to compute clustering")        

    def infer_transitions(self, it_nb=5000):
        """
        This method infers two quantities:
        - the SAMDP transition probability matrices given one skill
        - the agent transition probability matrix
        """

        if hasattr(self, 'cluster_coord'):

            cluster_nb = len(self.cluster_coord)
            skill_nb = len(self.options)
            SAMDP_transitions = np.zeros((skill_nb,cluster_nb,cluster_nb))
            agent_transitions = np.zeros((cluster_nb,cluster_nb))
            for it in range(it_nb):
                traj = self.generate_trajectory(T=100, noise=0.)
                for t in range(len(traj[0])): # every-visit MC
                    state, skill, next_state, reward = [traj[i][t] for i in range(4)]
                    cluster = self.clustering[state]
                    next_cluster = self.clustering[next_state]
                    SAMDP_transitions[skill,cluster,next_cluster] += 1
                    agent_transitions[cluster,next_cluster] += 1

            # normalization
            for skill in range(skill_nb):
                for cluster in range(cluster_nb):
                    SAMDP_sample_nb = SAMDP_transitions[skill,cluster,:].sum()
                    if SAMDP_sample_nb:
                        SAMDP_transitions[skill,cluster,:] /= float(SAMDP_sample_nb)
            for cluster in range(cluster_nb):
                agent_sample_nb = agent_transitions[cluster,:].sum()
                if agent_sample_nb:
                    agent_transitions[cluster,:] /= float(agent_sample_nb)

            self.SAMDP_transitions = SAMDP_transitions
            self.agent_transitions = agent_transitions

        else:
            print("Error : first run method 'aggregate_states' to compute clustering")            

            
def trainer(grid, dump_name=None, iovi_iters=200, option_updates=70, horizon=15, epsilon=0., noise=0., monitor=False,
    terminal_positions = [(0, 0), (0, 11), (8, 2), (5, 7)]):

    example = option_gridworld(grid=grid, terminal_positions=terminal_positions, noise=noise)
    example.IOVI(vi_steps=iovi_iters, option_updates=option_updates, horizon=horizon, epsilon=epsilon,
                 monitor_performance=monitor)
    if dump_name is not None:
        pickle.dump(example, open(dump_name + 'goodboy.pkl', 'wb'))
    return example

