import torch
import torch.nn.functional as F

import timeit
import numpy as np
import pandas as pd
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
    ep_lens = np.arange(250, 5001, 250)
    device = "cpu"

    # initialize random test values
    rewards = [torch.rand(n).to(device) for n in ep_lens]
    values = [torch.rand(n).to(device) for n in ep_lens]
    final_value = torch.rand(1).to(device)

    # compute GAE estimates
    conv_times = np.array([timeit.repeat(lambda: gae_estimate_conv(r, v, final_value), number=10) for r, v in zip(rewards, values)])
    loop_times = np.array([timeit.repeat(lambda: gae_estimate_loop(r, v, final_value), number=10) for r, v in zip(rewards, values)])

    # store results in DataFrame
    conv_times_flattened = conv_times.ravel()
    loop_times_flattened = loop_times.ravel()

    df = pd.DataFrame({
        "Episode Length": np.tile(np.repeat(ep_lens, 5), 2),
        "Time (s)": np.concatenate([conv_times_flattened, loop_times_flattened]),
        "Method": ["conv"] * (len(ep_lens) * 5) + ["loop"] * (len(ep_lens) * 5)
    })

    sns.lineplot(data=df, x="Episode Length", y="Time (s)", hue="Method")

if __name__ == "__main__":
    test_gae_estimate()
    plt.show()