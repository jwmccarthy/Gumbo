import torch as th


class Collector:

    def __init__(self, env, agent, buffer):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    @th.no_grad()
    def collect(self, steps=None, sample=True):
        last_obs, infos = self.env.reset()

        steps = steps or self.buffer.size

        for t in range(steps):
            actions = self.agent(last_obs, sample=sample)
            next_obs, rew, term, trunc, infos = self.env.step(actions)

            # truncated if last step and not terminal
            trunc |= (t == steps - 1) and ~term
            for i in range(self.env.num_envs):
                if trunc[i]: infos[i]["final_observation"] = next_obs[i]

            self.buffer.add(last_obs, actions, rew, term, trunc, infos)

            last_obs = next_obs

        return self.buffer