import torch
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

#unity_env = UnityEnvironment("D:\Practice\SentisInfrence\Build\SentisInfrence.exe", no_graphics=False)
#env = UnityToGymWrapper(unity_env)
env = gym.make("MountainCarContinuous-v0", render_mode="human")


model = torch.jit.load("models\\MountainCarContinuous_test_final.pth")
model.cpu()
model.eval()

hidden = (torch.zeros((1, 256), dtype=torch.float), torch.zeros((1, 256), dtype=torch.float))

with torch.inference_mode():
    for i in range(5):
        observation, _ = env.reset()
        state = torch.tensor(np.array([observation]), dtype=torch.float32)
        history_obs = np.zeros((1, env.observation_space.shape[0] * 64))
        history_actions = np.zeros((1, env.action_space.shape[0] * 64))
        done = False
        score = 0
        env.render()
        while not done:
            loc, scale_log, _ = model(state, torch.tensor(history_actions, dtype=torch.float), torch.tensor(history_obs, dtype=torch.float), hidden)
            scale = scale_log.exp()
            dist = Normal(loc, scale)
            sample = dist.sample()
            action = torch.tanh(sample)*torch.tensor(env.action_space.high)
            history_obs = np.append(history_obs, state, axis=1)
            history_obs = np.delete(history_obs, range(state.shape[-1]), axis=1)
            history_actions = np.append(history_actions, state, axis=1)
            history_actions = np.delete(history_actions, range(state.shape[-1]), axis=1)
            observation, r, terminated, truncated, _ = env.step(action.numpy()[0])
            score += r
            done = terminated or truncated
            state = torch.tensor(np.array([observation]), dtype=torch.float32)
        print(score)
env.close()