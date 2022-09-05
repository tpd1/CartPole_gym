import numpy as np
import gym
import pygame
import time
import matplotlib.pyplot as plt


def setup_env():
    env = gym.make("CartPole-v1")
    env.reset()
    for _ in range(100):
        env.render()  # Render on the screen
        action = env.action_space.sample()  # chose a random action
        env.step(action)  # Perform random action on the environment
        time.sleep(0.01)
    env.close()


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

# setup_env()
