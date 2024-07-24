import torch
import torch.nn.functional as F

import timeit
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

lmbda = 0.99
gamma = 0.95

def gae_estimate_conv(rewards, values, final_value):
    T = len(rewards)

    # compute TD errors
    next_vals = torch.cat([values[1:], final_value])
    td_errors = (rewards + gamma * next_vals - values).view(1, 1, -1)

    # conv kernel of discount factors
    kernel = (gamma * lmbda) ** torch.arange(T, dtype=torch.float32).view(1, 1, -1).to(td_errors.device)

    advantages = F.conv1d(td_errors, kernel, padding=T - 1).view(-1)[-T:]

    return advantages

def gae_estimate_loop(rewards, values, final_value):
    next_vals = torch.cat([values[1:], final_value])
    td_errors = rewards + gamma * next_vals - values
    advantages = torch.empty_like(td_errors).to(td_errors.device)

    adv = 0
    for i in reversed(range(len(rewards))):
        adv = td_errors[i] + lmbda * gamma * adv
        advantages[i] = adv

    return advantages

def test_gae_estimate():
    ep_lens = [100, 1000, 10000, 100000]
    device = "cpu"

    # initialize random test values
    rewards = [torch.rand(n).to(device) for n in ep_lens]
    values = [torch.rand(n).to(device) for n in ep_lens]
    final_value = torch.rand(1).to(device)

    # compute GAE estimates
    conv_times = [timeit.repeat(lambda: gae_estimate_conv(r, v, final_value), number=10) for r, v in zip(rewards, values)]
    loop_times = [timeit.repeat(lambda: gae_estimate_loop(r, v, final_value), number=10) for r, v in zip(rewards, values)]

    conv_time_mean = [np.mean(t) for t in conv_times]
    loop_time_mean = [np.mean(t) for t in loop_times]
    conv_time_std = [np.std(t) for t in conv_times]
    loop_time_std = [np.std(t) for t in loop_times]

    print(conv_time_mean, conv_time_std)
    print(loop_time_mean, loop_time_std)

if __name__ == "__main__":
    test_gae_estimate()
    plt.show()