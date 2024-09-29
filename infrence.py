import torch
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym

env = gym.make("HalfCheetah-v4", render_mode="human")

model = torch.jit.load("models\cheetah_final.pth")
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
            loc, scale = model(state)
            dist = Normal(loc, scale)
            action = dist.sample()
            observation, r, terminated, truncated, _ = env.step(action.numpy()[0])
            score += r
            done = terminated or truncated
            state = torch.tensor(np.array([observation]), dtype=torch.float32)
        print(score)
env.close()