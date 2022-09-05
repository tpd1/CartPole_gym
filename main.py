import numpy as np
import gym
import pygame
import time
import matplotlib.pyplot as plt


def create_bins(num_bins_per_action=10):
    bins_cart_pos = np.linspace(-4.8, 4.8, num_bins_per_action)
    bins_cart_velocity = np.linspace(-5, 5, num_bins_per_action)
    bins_pole_angle = np.linspace(-0.418, 0.418, num_bins_per_action)
    bins_pole_ang_vel = np.linspace(-5, 5, num_bins_per_action)

    return np.array([bins_cart_pos, bins_cart_velocity, bins_pole_angle, bins_pole_ang_vel])


NUM_BINS = 10
BINS = create_bins(NUM_BINS)  # Convert continuous space into discrete bins.


def discretize_observation(observations, bins):
    binned_observations = []
    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(observation, bins[i])
        binned_observations.append(discretized_observation)
    return tuple(binned_observations)  # Important for later indexing


def create_env():
    env = gym.make("CartPole-v1", render_mode='human')
    q_table_shape = (NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n) # setup_env()
    q_table = np.zeros(q_table_shape)
    print(q_table.shape)
    env.reset()


create_env()

