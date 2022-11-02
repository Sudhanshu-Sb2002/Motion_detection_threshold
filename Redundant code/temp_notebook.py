import matplotlib
import numpy as np
import os
from numba import njit, prange
import matplotlib.pyplot as plt

step = 0.001
v_array = np.arange(0, 0.06, step)
for i in range(len(v_array)):
    v_array[i] = round(v_array[i], 3)

v_dict = {i: v for i, v in enumerate(v_array)}
inverse_v_dict = {v: i for i, v in enumerate(v_array)}


# %%


@njit(parallel=True)
def block_designer():
    block_design = np.zeros(3)
    non_zero = np.array([-1, 1])
    all_stim = np.array([-1, 0, 1])
    # stim types is -1 0 or 1
    # we choose first stim randomly between -1 and 1, the rest are completely random among  -1 0 and 1
    block_design[0] = np.random.choice(non_zero)
    block_design[1] = np.random.choice(all_stim)
    if block_design[1] == 0:
        block_design[2] = np.random.choice(non_zero)
    else:
        block_design[2] = np.random.choice(all_stim)

    return np.random.permutation(block_design)


@njit(parallel=True)
def update_velocity(velocity, actual_responces, expected_responces, ANYLOSS=True):
    no_correct = int(np.sum(actual_responces == np.abs(expected_responces)))
    map_dictionary = {0: 0.003, 1: 0.003, 2: -0.002, 3: -0.002}

    if ANYLOSS == False:
        map_dictionary = {0: 0.005, 1: 0.005, 2: - 0.005, 3: -0.005}

    return round(velocity + map_dictionary[no_correct], 3)


@njit(parallel=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@njit(parallel=True)
def logistic_responder(mu, sigma, noise, start_velocity, iterations):
    probabilties = sigmoid((v_array - mu) / sigma)
    velocities = np.zeros(iterations)
    if probabilties[0.04 / step] < 0.8 or probabilties[0.02 / step] > 0.2:
        return velocities, -1

    velocities[0] = start_velocity
    responses = np.zeros(3)
    v = start_velocity
    zero_one = np.array([0, 1])
    p = 0
    for i in range(iterations - 1):
        block = block_designer()
        for j in prange(3):
            prob = probabilties[int(v / step)]
            prob = prob * (1 - noise) + np.random.uniform(0, 1) * (noise)
            p = np.random.uniform(0, 1)
            if p < prob:
                responses[j] = 1
            else:
                responses[j] = 0

        v = update_velocity(velocities[i], responses, block)
        velocities[i + 1] = v
        if v > 0.05 or v < 0.01:
            return velocities, -1
    return velocities, probabilties[int(v / step)]


@njit(parallel=True)
def simulate(mu_spread, sigma_spread, noise_spread, start_velocity, iterations):
    n_agents = mu_spread.shape[0] * sigma_spread.shape[0] * noise_spread.shape[0]
    velocities = np.zeros((n_agents, iterations))
    probabilities = np.zeros(n_agents)

    for i in range(mu_spread.shape[0]):
        if i % 5 == 0:
            print(i / mu_spread.shape[0])
        for j in prange(sigma_spread.shape[0]):
            for k in prange(noise_spread.shape[0]):
                index = i * sigma_spread.shape[0] * noise_spread.shape[0] + j * noise_spread.shape[0] + k
                velocities[index], probabilities[index] = logistic_responder(mu_spread[i], sigma_spread[j],
                                                                             noise_spread[k], start_velocity,
                                                                             iterations)
    return velocities, probabilities


# %%
mu_spread = np.random.uniform(0.02, 0.04, 40)
sigma_spread = np.random.uniform(0.004, 0.002, 40)
intrinsic_noise_spread = np.array([0, 0.001, 0.005, 0.01, 0.05, 0.1])

iterations = 100
start_velocity = 0.04

velocities, probabilities = simulate(mu_spread, sigma_spread, intrinsic_noise_spread, start_velocity, iterations)
converged_indices = np.where(probabilities != -1)[0]
converged_velocities = velocities[converged_indices]
converged_probabilities = probabilities[converged_indices]

# %%
converged_muspread = np.zeros(converged_indices.shape[0])
converged_sigmaspread = np.zeros(converged_indices.shape[0])
converged_noisespread = np.zeros(converged_indices.shape[0])
for i in range(converged_indices.shape[0]):
    converged_muspread[i] = mu_spread[converged_indices[i] // (sigma_spread.shape[0] * intrinsic_noise_spread.shape[0])]
    converged_sigmaspread[i] = sigma_spread[
        (converged_indices[i] // intrinsic_noise_spread.shape[0]) % sigma_spread.shape[0]]
    converged_noisespread[i] = intrinsic_noise_spread[converged_indices[i] % intrinsic_noise_spread.shape[0]]
print(converged_muspread[0], converged_sigmaspread[0], converged_noisespread[0])

# %%
i = 2
print(converged_probabilities[i])
plt.plot(converged_velocities[i])
plt.show()
# %%

mu_colors = (converged_muspread - converged_muspread.min()) / (converged_muspread.max() - converged_muspread.min())
sigma_colors = (converged_sigmaspread - converged_sigmaspread.min()) / (
            converged_sigmaspread.max() - converged_sigmaspread.min())
noise_colors = (converged_noisespread - converged_noisespread.min()) / (
            converged_noisespread.max() - converged_noisespread.min())
colors = np.stack((mu_colors, sigma_colors, noise_colors), axis=1)

plt.scatter(np.arange(len(converged_indices)), converged_probabilities, c=colors)
# %%
plt.hist(converged_probabilities)