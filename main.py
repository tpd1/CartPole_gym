import numpy as np
import gym
import pygame
import time
import matplotlib.pyplot as plt

plt.interactive(False)

EPOCHS = 25000
ALPHA = 0.6
GAMMA = 1
NUM_BINS = 25

MAX_EPSILON = 1.0  # Exploration probability at start
MIN_EPSILON = 0.001  # Minimum exploration probability
DECAY_RATE = 0.00025

BURN_IN = 1
EPSILON_END = 22000
EPSILON_REDUCE = 0.00025


def create_bins(num_bins_per_action=10):
    bins_cart_pos = np.linspace(-4.8, 4.8, num_bins_per_action)
    bins_cart_velocity = np.linspace(-5, 5, num_bins_per_action)
    bins_pole_angle = np.linspace(-0.418, 0.418, num_bins_per_action)
    bins_pole_ang_vel = np.linspace(-5, 5, num_bins_per_action)

    return np.array([bins_cart_pos, bins_cart_velocity, bins_pole_angle, bins_pole_ang_vel])


BINS = create_bins(NUM_BINS)  # Convert continuous space into discrete bins.


def discretize_observation(observations, bins):
    binned_observations = []
    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(observation, bins[i])
        binned_observations.append(discretized_observation)
    return tuple(binned_observations)  # Important for later indexing


def epsilon_greedy_action_selection(env, epsilon, q_table, discrete_state):
    '''
    Returns an action for the agent. Note how it uses a random number to decide on
    exploration versus explotation trade-off.
    '''
    random_number = np.random.random()
    # EXPLOITATION, USE BEST Q(s,a) Value
    if random_number > epsilon:
        action = np.argmax(q_table[discrete_state])
    # EXPLORATION, USE A RANDOM ACTION
    else:
        action = np.random.randint(0, env.action_space.n)

    return action


def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)


def reduce_epsilon(epsilon, epoch):
    '''
    Linear reduction of epsilon in the first 10,000 epochs
    '''
    if BURN_IN <= epoch <= EPSILON_END:
        epsilon -= EPSILON_REDUCE
    return epsilon


def fail(done, points, reward):
    if done and points < 400:
        reward = -450
    return reward


def create_env():
    env = gym.make("CartPole-v1")
    q_table_shape = (NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n)  # setup_env()
    q_table = np.zeros(q_table_shape)
    env.reset()
    ##############################################
    ### VISUALIZATION OF TRAINING PROGRESS ######
    #############################################

    log_interval = 1000  # How often do we update the plot? (Just for performance reasons)
    render_interval = 2000  # How often to render the game during training (If you want to watch your model learning)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    fig.canvas.draw()
    ##############################################
    #############################################

    points_log = []  # to store all achieved points
    mean_points_log = []  # to store a running mean of the last 30 results
    epochs = []  # store the epoch for plotting
    epsilon = 1

    for epoch in range(EPOCHS):
        initial_state = env.reset()
        discretized_state = discretize_observation(initial_state, BINS)
        done = False
        points = 0
        epochs.append(epoch)

        while not done:
            if epoch % render_interval == 0:
                env.render()

            action = epsilon_greedy_action_selection(env, epsilon, q_table, discretized_state)
            next_state, reward, done, info = env.step(action)
            # reward = fail(done, points, reward)

            next_state_discretized = discretize_observation(next_state, BINS)
            old_q_value = q_table[discretized_state + (action,)]
            next_optimal_q_value = np.max(q_table[next_state_discretized])
            next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)

            q_table[discretized_state + (action,)] = next_q  # Insert next Q-Value into the table
            discretized_state = next_state_discretized  # Update the old state
            points += 1

        epsilon = reduce_epsilon(epsilon, epoch)  # Reduce epsilon
        points_log.append(points)  # log overall achieved points for the current epoch
        running_mean = round(np.mean(points_log[-30:]), 2)  # Compute running mean points over the last 30 epochs
        mean_points_log.append(running_mean)  # and log it

        if epoch % log_interval == 0:
            ax.clear()
            # ax.scatter(epochs, points_log)
            ax.plot(epochs, points_log)
            ax.plot(epochs, mean_points_log, label=f"Running Mean: {running_mean}")
            plt.legend()
            plt.pause(0.0001)
            fig.canvas.draw()
            plt.show()

    env.close()


create_env()
