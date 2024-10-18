import torch
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

unity_env = UnityEnvironment("D:\Practice\AIProject\Build\AIProject.exe", no_graphics=False)
env = UnityToGymWrapper(unity_env)
#env = gym.make("MountainCarContinuous-v0", render_mode="human")


model = torch.jit.load("models\\cart_test_final.pth")
model.cpu()
model.eval()

with torch.inference_mode():
    for i in range(3):
        observation = env.reset()
        state = torch.tensor(np.array([observation]), dtype=torch.float32)
        history = np.zeros((1, (env.observation_space.shape[0] + env.action_space.shape[0]) * 128))
        done = False
        score = 0
        env.render()
        while not done:
            loc, scale_log = model(state, torch.tensor(history, dtype=torch.float))
            scale = scale_log.exp()
            dist = Normal(loc, scale)
            sample = dist.sample()
            action = torch.tanh(sample)*torch.tensor(env.action_space.high)
            t = torch.cat((state, action), dim=1)
            history = np.append(history, t, axis=1)
            history = np.delete(history, range(t.shape[-1]), axis=1)
            observation, r, terminated, _ = env.step(action.numpy()[0])
            score += r
            done = terminated
            state = torch.tensor(np.array([observation]), dtype=torch.float32)
        print(score)
env.close()