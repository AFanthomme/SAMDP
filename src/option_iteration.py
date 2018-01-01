# Implementation of the Time-Regularized Interrupting options framework
# Ref : Mankowitz, D. J.; Mann, T. A.; and Mannor, S. 2014. Time regularized interrupting options. International
#       Conference on Machine Learning

# Implementation : Arnaud Fanthomme

import numpy as np
from src.gridworld import GridWorld
import matplotlib.pyplot as p
import src.gridrender as gui


class decorated_env:
    def __init__(self, env):
        self.env = env
        self.n_states = env.n_states
        # Default options are going straight in one direction, can be started anywhere and never terminate.
        self.n_options = 4
        self.options = [
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 0 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 1 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 2 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 3 * np.ones(self.n_states, dtype=int)}]


    def q_estimation(self, n_epochs=100, epsilon=0.1, seed=0):
        env = self.env
        n_actions = 4  # Hardcode this for simplicity as in ex_1
        np.random.seed(seed)
        ep_length = 30 # Set it high to be in the case where all trajectories reach exits
        nb_encounters = np.zeros((env.n_states, n_actions))
        q_estimate = np.zeros((env.n_states, n_actions))
        rewards_record = []

        # The learning rate for Q(x,a) at a given time t is taken as the inverse of
        # the number of times (x,a) has been visited previously.
        # This is the simplest rate satisfying the Robbins-Monro conditions.
        alpha = lambda state, action: 1. / nb_encounters[state, action]

        for epoch in range(n_epochs):
            t = 0
            state = env.reset()
            term = False
            epoch_reward = 0

            while not term and t < ep_length:
                possible_actions = env.state_actions[state]
                if np.random.rand() < 1. - epsilon:
                    # With probability 1-epsilon, act according to the greedy policy, estimated using Q
                    actions = np.argwhere(q_estimate[state, :] == np.max(q_estimate[state, :])).flatten()
                    actions = np.intersect1d(actions, possible_actions)
                else:
                    # With probability epsilon, pick an available action at random.
                    # This ensures that every state should be visited an infinite number of times.
                   actions = possible_actions

                action = actions[np.random.randint(len(actions))]
                nextstate, reward, term = env.step(state, action)

                nb_encounters[state, action] += 1
                epoch_reward += reward

                alpha_k = alpha(state, action)
                q_estimate[state, action] = (1. - alpha_k) * q_estimate[state, action] + alpha_k * (reward + env.gamma *
                                    np.max(q_estimate[nextstate, :]))
                state = nextstate
                t += 1
            rewards_record.append(epoch_reward)

        return q_estimate, rewards_record


    def TRIOVI(self, vi_steps=100, option_updates=100, regularizer=None):
        """
        Use the TRIOVI rule to train skills by interrrupting the initial options.

        :param initial_options:
        :param regularizer:
        :return:
        """

        if type(regularizer) == float:
            rho = lambda t: regularizer ** t * 1. / (1-0.95)
        else:
            rho = lambda t: 0.1

        initial_options = self.options
        current_options = initial_options
        alpha = np.ones((self.n_options, self.n_states))

        for t in range(option_updates):
            # Since we do not have access to the real transition probabilities, we use the Q-learning procedure from TD1
            # instead of VI to obtain the new Q function.
            current_Q, _ = self.q_estimation(n_epochs=vi_steps)
            # Q : n_states, n_actions
            for idx, option in enumerate(current_options):
                option['term'] = np.maximum(initial_options[idx]['term'],
                                        (current_Q[:, idx] < np.max(current_Q, axis=1) - rho(t) * alpha[idx, :]).astype(float))
                alpha[idx, :] = option['term'] < 1

        self.options = current_options
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
        possible_actions = [i for i in range(self.n_options) if state in self.options[i]['init']]
        return np.random.choice(possible_actions)

    def generate_trajectory(self, T=20):
        env = self.env
        state = env.reset()
        skill_in_use = self.choose_skill(state)
        termination_probs = self.options[skill_in_use]['term']
        trajectory = np.zeros(T, dtype=int)
        trajectory[0] = state
        rewards = np.zeros(T)
        rewards[0] = 0

        for t in range(1, T):
            nextstate, reward, term = env.step(state, skill_in_use)
            trajectory[t] = nextstate
            rewards[t] = reward
            state = nextstate

            if term:
                break

            if np.random.rand() < termination_probs[state]:
                # With probability termination_probs[state], interrupt option and choose a new one
                skill_in_use = self.choose_skill(state)
                termination_probs = self.options[skill_in_use]['term']

        return trajectory, rewards


def test_triovi():
    grid = \
        [
            ['', '', '', 1],
            ['', 'x', '', -1],
            ['', '', '', '']
        ]
    env = GridWorld(grid=grid, gamma=0.95)
    example = decorated_env(env)

    # Remove from inits states where a certain action cannot be initiated
    example.options[0]['init'] = np.delete(example.options[0]['init'], [3, 6, 10])
    example.options[1]['init'] = np.delete(example.options[1]['init'], [3, 6, 1, 7, 8, 9, 10])
    example.options[2]['init'] = np.delete(example.options[2]['init'], [0, 3, 6, 5, 7])
    example.options[3]['init'] = np.delete(example.options[3]['init'], [0, 1, 2, 3, 6, 8])

    # Add termination probabilities of 1 for states where a certain action is forbidden
    example.options[0]['term'][[3, 6, 10]] = 1.
    example.options[1]['term'][[1, 3, 6, 7, 8, 9, 10]] = 1.
    example.options[2]['term'][[0, 3, 6, 5, 7]] = 1.
    example.options[3]['term'][[0, 1, 2, 3, 6, 8]] = 1.

    example.TRIOVI(vi_steps=40, option_updates=100)
    example.plot_terminations()


if __name__ == '__main__':
    test_triovi()