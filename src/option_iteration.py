# Implementation of the Time-Regularized Interrupting options framework
# Ref : Mankowitz, D. J.; Mann, T. A.; and Mannor, S. 2014. Time regularized interrupting options. International
#       Conference on Machine Learning

# Implementation : Arnaud Fanthomme

import numpy as np
from src.gridworld import GridWorld
import matplotlib.pyplot as p
import src.gridrender as gui
from copy import deepcopy
import pickle, pdb
from progress.bar import Bar
from tkinter import Tk
import tkinter.font as tkfont
import numbers

class decorated_env:
    def __init__(self, env, terminal_states=None):
        self.env = env
        self.n_states = env.n_states
        self.terminal_states =terminal_states
        # Default options are going straight in one direction, can be started anywhere and never terminate.
        self.n_options = 4
        self.options = [
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 0 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 1 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 2 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 3 * np.ones(self.n_states, dtype=int)}]

        # Not functional
        # if terminal_states:
        #     for state in terminal_states:
        #         for option in range(self.n_options):
        #             self.options[option]['init'] = np.delete(self.options[option]['init'], state)
        #             self.options[option]['term'][state] = 1.


    def q_estimation(self, n_epochs=100, T=30, epsilon=0., seed=None):
        env = self.env
        n_actions = 4  # Hardcode this for simplicity as in ex_1
        if seed:
            np.random.seed(seed)
        ep_length = T # Set it high to be in the case where all trajectories reach exits
        nb_encounters = np.zeros((env.n_states, n_actions))
        q_estimate = np.zeros((env.n_states, n_actions))

        # The learning rate for Q(x,a) at a given time t is taken as the inverse of
        # the number of times (x,a) has been visited previously.
        # This is the simplest rate satisfying the Robbins-Monro conditions.
        alpha = lambda state, action: 1. / nb_encounters[state, action]

        for epoch in range(n_epochs):
            states, skills, next_states, rewards = self.generate_trajectory(T=ep_length, noise=epsilon)
            idx = 0
            for state, skill, next_state, reward in zip(states, skills, next_states, rewards):
                if idx == 0:
                    previous_option = skill
                    skill_reward = 0
                    initiating_state = state
                elif idx == len(states) - 1:
                    skill_reward += reward
                    nb_encounters[initiating_state, previous_option] += 1
                    alpha_k = alpha(initiating_state, previous_option)
                    q_estimate[initiating_state, previous_option] = (1. - alpha_k) * q_estimate[
                        initiating_state, previous_option] + alpha_k * \
                        (skill_reward + env.gamma * np.max(q_estimate[next_state, :]))

                elif previous_option != skill:
                    nb_encounters[initiating_state, previous_option] += 1
                    alpha_k = alpha(initiating_state, previous_option)
                    q_estimate[initiating_state, previous_option] = (1. - alpha_k) * q_estimate[initiating_state, previous_option] + alpha_k * \
                                            (skill_reward + env.gamma * np.max(q_estimate[next_state, :]))
                    previous_option = skill
                    skill_reward = reward
                    initiating_state = state
                else:
                    skill_reward += reward
                idx += 1


        return q_estimate


    def TRIOVI(self, vi_steps=100, option_updates=100, regularizer=0.95, horizon=30, monitor_performance=None, epsilon=0.):
        """
        Use the TRIOVI rule to train skills by interrrupting the initial options.

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
            if t%(option_updates//10) == 0:
                print('{}% done'.format((100.*t)//option_updates))
            # Since we do not have access to the real transition probabilities, we use the Q-learning procedure from TD1
            # instead of VI to obtain the new Q function.
            current_Q = self.q_estimation(n_epochs=vi_steps, T=horizon, epsilon=epsilon)

            for idx, option in enumerate(current_options):
                option['term'] = np.maximum(initial_options[idx]['term'],
                                        (current_Q[:, idx] < np.max(current_Q, axis=1) - regularizer * alpha[idx, :]).astype(float))
                alpha[idx, :] = option['term'] < 1

            if monitor_performance:
                performance_record[t] = np.mean([np.sum(self.generate_trajectory(T=20)[3]) for _ in range(monitor_performance)])

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


    def choose_skill(self, state):
        possible_actions = [i for i in range(self.n_options) if state in self.options[i]['init']
                             and self.options[i]['term'][state] < 0.99]
        try:
            return np.random.choice(possible_actions)
        except ValueError:
            raise RuntimeError('No actions possible in state {}'.format(state))

    def generate_trajectory(self, T=30, noise=0.):
        env = self.env
        state = env.reset()

        # Trajectories that directly land on a terminal state are not very useful
        while state in self.terminal_states:
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
                next_states = next_states[:t+1]
                states = states[:t+1]
                rewards = rewards[:t+1]
                skills = skills[:t+1]
                break
            elif np.random.rand() < termination_probs[state] or np.random.rand() < noise:
                # With probability termination_probs[state], interrupt option and choose a new one
                skill_in_use = self.choose_skill(state)
                termination_probs = self.options[skill_in_use]['term']

        return states, skills, next_states, rewards

    def aggregate_states(self, K, w, nb_it=100):
        # K : number of clusters
        # w : size of the sliding window

        ##### random initialization ######

        N = self.env.n_states
        cluster_coord = [ self.env.state2coord[state] for state in np.random.choice(range(N),size=K,replace=False) ]
        clustering = np.zeros(N, dtype=int)

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
            for state in range(N):
                if state in traj:
                    state_idx = np.where(traj==state)[0][0]
                    ext_state = traj[int(max(state_idx-w, 0)):int(min(state_idx+w+1, len(traj)))]
                    ext_state_coord = np.array([self.env.state2coord[i] for i in ext_state])
                    clustering[state] = np.argmin([np.linalg.norm(ext_state_coord-cluster_coord[k]) for k in range(K)])

        self.cluster_coord = cluster_coord
        self.clustering = clustering

    def show_clustering(self):

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

            cluster_color = []
            r = lambda: np.random.randint(0,255)
            for cluster_idx in range(len(self.cluster_coord)):
                cluster_color.append('#%02X%02X%02X' % (r(),r(),r()))

            self.polygons = []
            for state, cluster_idx in enumerate(self.clustering):
                r, c = self.env.state2coord[state]
                x, y = 10 + c * (dim + 4), 10 + r * (dim + 4)
                self.polygons.append( self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black', fill=cluster_color[cluster_idx], width=2) )

            self.ovals = []
            for cluster_idx,cluster in enumerate(self.cluster_coord):
                r1, c1 = 10 + cluster[1] * (dim + 4), 10 + cluster[0] * (dim + 4)
                x1, y1 = r1 + dim / 2., c1 + dim / 2.
                self.ovals.append( self.window.create_oval(x1 - dim / 5., y1 - dim / 5., x1 + dim / 5., y1 + dim / 5., fill=cluster_color[cluster_idx]) )

            for oval in self.ovals:
                if hasattr(self, str(oval)):
                    self.window.delete(oval)
            for polygon in self.polygons:
                if hasattr(self, str(polygon)):
                    self.window.delete(polygon)

            self.window.update()
        else:
            print("Error : first run method 'aggregate_states' to compute clustering")        

def test_triovi():
    grid = \
        [
            ['', '', '', 1],
            ['', 'x', '', -1],
            ['', '', '', '']
        ]
    env = GridWorld(grid=grid, gamma=0.95, time_penalty=0.0)
    example = decorated_env(env, terminal_states=[3, 6])

    # Remove from inits states where a certain action cannot be initiated
    example.options[0]['init'] = np.delete(example.options[0]['init'], [3, 4, 6, 10])
    example.options[1]['init'] = np.delete(example.options[1]['init'], [1, 3, 6, 7, 8, 9, 10])
    example.options[2]['init'] = np.delete(example.options[2]['init'], [0, 3, 4, 6, 5, 7])
    example.options[3]['init'] = np.delete(example.options[3]['init'], [0, 1, 2, 3, 6, 8])

    # Add termination probabilities of 1 for states where a certain action is forbidden
    example.options[0]['term'][[3, 4, 6, 10]] = 1.
    example.options[1]['term'][[1, 3, 6, 7, 8, 9, 10]] = 1.
    example.options[2]['term'][[0, 3, 4, 6, 5, 7]] = 1.
    example.options[3]['term'][[0, 1, 2, 3, 6, 8]] = 1.

    example.TRIOVI(vi_steps=50, option_updates=40, monitor_performance=100, regularizer=0.95)
    # example.plot_terminations()
    example.env.render = True
    for _ in range(5):
        example.generate_trajectory()

def test_true_grid():
    grid = \
        [
            ['', '', '', -1, 'x', 1, 'x'],
            ['', '', 'x', 'x', '', '', ''],
            ['', '', '', '', '', 'x', ''],
            [1, 'x', '', '', '', '', ''],
        ]
    env = GridWorld(grid=grid, gamma=0.95, time_penalty=0.0, noise=0.1)
    example = decorated_env(env, terminal_states=[3, 4, 16])

    # Remove from inits states where a certain action cannot be initiated
    example.options[0]['init'] = np.delete(example.options[0]['init'], [3, 4, 6, 9, 14, 15, 16, 21])
    example.options[1]['init'] = np.delete(example.options[1]['init'], [2, 3, 4, 8, 11, 16, 17, 18, 19, 20, 21])
    example.options[2]['init'] = np.delete(example.options[2]['init'], [0, 3, 4, 5, 7, 10, 15, 16, 17])
    example.options[3]['init'] = np.delete(example.options[3]['init'], [0, 1, 2, 3, 4, 7, 9, 12, 13, 16, 20])

    # Add termination probabilities of 1 for states where a certain action is forbidden
    example.options[0]['term'][[3, 4, 6, 9, 14, 15, 16, 21]] = 1.
    example.options[1]['term'][[2, 3, 4, 8, 11, 16, 17, 18, 19, 20, 21]] = 1.
    example.options[2]['term'][[0, 3, 4, 5, 7, 10, 15, 16, 17]] = 1.
    example.options[3]['term'][[0, 1, 2, 3, 4, 7, 9, 12, 13, 16, 20]] = 1.

    example.TRIOVI(vi_steps=4500, option_updates=100, horizon=20, monitor_performance=300, regularizer=0.)
    pickle.dump(example, open('grid1.pkl', 'wb'))
    example.plot_terminations()
    example.env.noise = 0.
    """example.env.render = True
    for _ in range(25):
        example.generate_trajectory()"""

    return example

def gridmaker(grid, terminal_positions):
    env = GridWorld(grid=grid, gamma=0.95, time_penalty=0.0, noise=0.1)
    # print(env.state2coord)
    terminal_states = [env.coord2state[position[0], position[1]] for position in terminal_positions]
    options_mdp = decorated_env(env, terminal_states=terminal_states)
    problematic_states = [[state for state in range(env.n_states) if action not in env.state_actions[state] or
                           state in terminal_states] for action in range(4)]

    # Remove from inits states where a certain action cannot be initiated
    for action in range(4):
        options_mdp.options[action]['init'] = np.delete(options_mdp.options[action]['init'], problematic_states[action])

    # Add termination probabilities of 1 for states where a certain action is forbidden
    for action in range(4):
        options_mdp.options[action]['term'][problematic_states[action]] = 1.

    return options_mdp

def test_gridmaker():
    grid = \
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

    terminal_positions = [(0,0), (0, 11), (2, 8), (5, 7)]
    example = gridmaker(grid, terminal_positions)
    example.TRIOVI(vi_steps=4000, option_updates=70, horizon=15, monitor_performance=None, regularizer=0.)
    pickle.dump(example, open('goodboy.pkl', 'wb'))
    example.plot_terminations()
    example.env.noise = 0.
    """example.env.render = True
    for _ in range(25):
        example.generate_trajectory()"""

    return example

if __name__ == '__main__':
    # test_triovi()
    # test_true_grid()
    test_gridmaker()
