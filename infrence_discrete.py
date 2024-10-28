import torch
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

#unity_env = UnityEnvironment("D:\Practice\SentisInfrence\Build\SentisInfrence.exe", no_graphics=False)
#env = UnityToGymWrapper(unity_env)
env = gym.make("CartPole-v1", render_mode="human")


model = torch.jit.load("models\\unity_test_final.pth")
model.cpu()
model.eval()

with torch.inference_mode():
    for i in range(5):
        observation, _ = env.reset()
        state = torch.tensor(np.array([observation]), dtype=torch.float32)
        done = False
        score = 0
        env.render()
        while not done:
            probs = model(state)
            dist = Categorical(probs)
            action = dist.sample()
            observation, r, terminated, truncated, _ = env.step(action.numpy()[0])
            score += r
            done = terminated or truncated
            state = torch.tensor(np.array([observation]), dtype=torch.float32)
        print(score)
env.close()