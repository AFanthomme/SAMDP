# Implementation of the Time-Regularized Interrupting options framework
# Ref : Mankowitz, D. J.; Mann, T. A.; and Mannor, S. 2014. Time regularized interrupting options. International
#       Conference on Machine Learning

# Implementation : Arnaud Fanthomme

import numpy as np
from src.gridworld import GridWorld
import matplotlib.pyplot as p
import src.gridrender as gui


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


    def q_estimation(self, n_epochs=100, epsilon=0., seed=None):
        env = self.env
        n_actions = 4  # Hardcode this for simplicity as in ex_1
        if seed:
            np.random.seed(seed)
        ep_length = 30 # Set it high to be in the case where all trajectories reach exits
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


    def TRIOVI(self, vi_steps=100, option_updates=100, regularizer=None, monitor_performance=None):
        """
        Use the TRIOVI rule to train skills by interrrupting the initial options.

        :param initial_options:
        :param regularizer:
        :return:
        """

        if type(regularizer) == float:
            rho = lambda t: regularizer ** t * 1. / (1-0.95)
        else:
            rho = lambda t: 0.95

        if monitor_performance:
            performance_record = np.zeros(option_updates)

        initial_options = self.options
        current_options = initial_options
        alpha = np.ones((self.n_options, self.n_states))

        for t in range(option_updates):
            # Since we do not have access to the real transition probabilities, we use the Q-learning procedure from TD1
            # instead of VI to obtain the new Q function.
            current_Q = self.q_estimation(n_epochs=vi_steps)

            for idx, option in enumerate(current_options):
                option['term'] = np.maximum(initial_options[idx]['term'],
                                        (current_Q[:, idx] < np.max(current_Q, axis=1) - rho(t) * alpha[idx, :]).astype(float))
                alpha[idx, :] = option['term'] < 1

            if monitor_performance:
                performance_record[t] = np.mean([np.sum(self.generate_trajectory(T=20)[3]) for _ in range(monitor_performance)])

        if monitor_performance:
            p.plot(performance_record)
            p.title('Evolution of the performance as a function of the number of iterations')
            p.show()

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
        possible_actions = [i for i in range(self.n_options) if state in self.options[i]['init']
                             and self.options[i]['term'][state] < 0.99]
        return np.random.choice(possible_actions)

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

    example.TRIOVI(vi_steps=50, option_updates=40, monitor_performance=100)
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

    example.TRIOVI(vi_steps=50, option_updates=100, monitor_performance=400, regularizer=2)
    example.plot_terminations()
    example.env.noise = 0.
    example.env.render = True
    for _ in range(25):
        example.generate_trajectory()

if __name__ == '__main__':
    # test_triovi()
    test_true_grid()