import torch
import torch.nn.functional as F
from torchaudio.functional import fftconvolve

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

def gae_estimate_fft_conv(rewards, values, final_value):
    T = len(rewards)

    # compute TD errors
    next_vals = torch.cat([values[1:], final_value])
    td_errors = rewards + gamma * next_vals - values

    # conv kernel of discount factors
    kernel = (gamma * lmbda) ** torch.arange(T, dtype=torch.float32).to(td_errors.device)

    advantages = fftconvolve(td_errors, kernel.flip(0))

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
    n = 10
    ep_lens = np.array([100, 1000, 10000, 100000])
    device = "cuda"

    # initialize random test values
    rewards = [torch.rand(n).to(device) for n in ep_lens]
    values = [torch.rand(n).to(device) for n in ep_lens]
    final_value = torch.rand(1).to(device)

    # compute GAE estimates
    conv_times = []
    for r, v in zip(rewards, values):
        for _ in range(10): gae_estimate_conv(r, v, final_value) # warm up
        conv_times.append(timeit.repeat(lambda: gae_estimate_conv(r, v, final_value), number=n))
    conv_times = np.array(conv_times)

    conv_fft_g = []
    for r, v in zip(rewards, values):
        for _ in range(10): gae_estimate_fft_conv(r, v, final_value) # warm up
        conv_fft_g.append(timeit.repeat(lambda: gae_estimate_fft_conv(r, v, final_value), number=n))
    conv_fft_g = np.array(conv_fft_g)

    device = "cpu"

    # initialize random test values
    rewards = [torch.rand(n).to(device) for n in ep_lens]
    values = [torch.rand(n).to(device) for n in ep_lens]
    final_value = torch.rand(1).to(device)

    loop_times = np.array([timeit.repeat(lambda: gae_estimate_loop(r, v, final_value), number=n) for r, v in zip(rewards, values)])
    conv_cpu_t = np.array([timeit.repeat(lambda: gae_estimate_conv(r, v, final_value), number=n) for r, v in zip(rewards, values)])
    conv_fft_t = np.array([timeit.repeat(lambda: gae_estimate_fft_conv(r, v, final_value), number=n) for r, v in zip(rewards, values)])

    # store results in DataFrame
    conv_times_flattened = conv_times.ravel()
    loop_times_flattened = loop_times.ravel()
    conv_cpu_t_flattened = conv_cpu_t.ravel()
    conv_fft_t_flattened = conv_fft_t.ravel()
    conv_fft_g_flattened = conv_fft_g.ravel()

    df = pd.DataFrame({
        "Episode Length": np.tile(np.repeat(ep_lens, n/2), 5),
        "Time (s)": np.concatenate([loop_times_flattened,
                                    conv_cpu_t_flattened,
                                    conv_times_flattened,
                                    conv_fft_t_flattened,
                                    conv_fft_g_flattened]), 
        "Method": ["Loop"] * (len(ep_lens) * int(n/2))
                + ["conv1d (CPU)"] * (len(ep_lens) * int(n/2)) \
                + ["conv1d (GPU)"] * (len(ep_lens) * int(n/2)) \
                + ["fftconvolve (CPU)"] * (len(ep_lens) * int(n/2)) \
                + ["fftconvolve (GPU)"] * (len(ep_lens) * int(n/2)) \
    })

    plt.figure(figsize=(8, 6))
    sns.set_theme()
    g = sns.lineplot(data=df, x="Episode Length", y="Time (s)", hue="Method")
    g.set_xscale("log")
    g.set_yscale("log")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_gae_estimate()
    plt.show()