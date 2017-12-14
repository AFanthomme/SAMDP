# Implementation of the Time-Regularized Interrupting options framework
# Ref : Mankowitz, D. J.; Mann, T. A.; and Mannor, S. 2014. Time regularized interrupting options. International
#       Conference on Machine Learning

# Implementation : Arnaud Fanthomme

import numpy as np


class decorated_env:
    def __init__(self, env):
        self.env = env
        self.n_states = env.n_states
        # Initial options are going straight in one direction, can be started anywhere and never terminate.
        self.options = [
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 0 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 1 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 2 * np.ones(self.n_states, dtype=int)},
        {'init': range(self.n_states), 'term': np.zeros(self.n_states), 'pol': 3 * np.ones(self.n_states, dtype=int)}]


def value_iteration(policy, Q):
    return Q


def policy_optimization(env, n_epochs=100, epsilon=0.1, seed=0):
    n_actions = 4  # Hardcode this for simplicity as in ex_1
    np.random.seed(seed)
    ep_length = 30 # Set it high to be in the case where all trajectories reach exits
    nb_encounters = np.zeros((env.n_states, n_actions))
    q_estimate = np.zeros((env.n_states, n_actions))
    rewards_record = []
    deviations_record = []

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



def TRIOVI(decorated_env, vi_steps=100, option_updates=100, regularizer=None):
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

    initial_options = decorated_env.options
    current_options = initial_options
    alpha = np.ones((len(decorated_env.options), decorated_env.n_states))

    for t in range(option_updates):
        # Since we do not have access to the real transition probabilities, we use the Q-learning procedure from TD1
        # instead of VI to obtain the new Q function.
        current_Q = policy_optimization(decorated_env.env, n_epochs=vi_steps)
        # Q : n_states, n_actions
        for idx, option in enumerate(current_options):
            option['term'] = np.max(initial_options[idx]['term'],
                    (current_Q < np.max(current_Q, axis=1) - rho(t) * alpha).astype(float))
            alpha[idx, :] = option['term'] < 1

    return current_options