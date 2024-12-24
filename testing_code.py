import numpy as np
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.pool import ThreadPool
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.categorical import Categorical
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import multiprocessing as mp

device = (torch.device("cuda"))
dim = 128
critic_lr = 3e-4
actor_lr = critic_lr / 3.0
reparam_noise = 1e-6
gamma=0.99
tau=0.005
alpha_start = 1
max_size = 1000000
batch_size = 64
total_plays = 50000
num_epochs = 1
N = 100

# unity_env = UnityEnvironment("D:\Practice\SentisInfrence\Build\SentisInfrence.exe", no_graphics=True)
# env = UnityToGymWrapper(unity_env)

env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape
n_actions=env.action_space.n

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.done_memory = np.zeros((self.mem_size), dtype=bool)

    def store_transition(self, state, action, reward, state_, dones):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = dones

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, states_, dones
    
class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_dim[-1], dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=0, total_iters=total_plays)

        self.to(device)

    def forward(self, state):
        q = self.layers(state)

        return q

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(*obs_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, int(n_actions)),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=0, total_iters=total_plays)

        self.to(device)

    def forward(self, state):
        action_probs = self.layers(state)
       

        return action_probs

    def sample_normal(self, state):
        action_probs = self.forward(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()

        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(action_probs + z)

        return action, log_probs, action_probs
    
class Agent():
    def __init__(self):
        self.memory = ReplayBuffer(max_size, obs_dim, n_actions)

        self.actor = ActorNetwork()
        self.critic_1 = CriticNetwork()
        self.critic_2 = CriticNetwork()
        self.critic_1_target = CriticNetwork()
        self.critic_2_target = CriticNetwork()

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.alpha = alpha_start
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_lr)
        self.alpha_scheduler = optim.lr_scheduler.LinearLR(self.alpha_optimizer, start_factor=1, end_factor=0, total_iters=total_plays)
        self.target_entropy = -n_actions

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation], dtype=float), dtype=torch.float).to(device)
        actions, _, _ = self.actor.sample_normal(state)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, dones):
        self.memory.store_transition(state, action, reward, new_state, dones)

    def gradient_step(self):
        if self.memory.mem_cntr < batch_size:
            return
        
        for _ in range(num_epochs):
            state, actions, reward, state_, dones = self.memory.sample_buffer(batch_size)

            reward = torch.tensor(reward, dtype=torch.float).to(device)
            state_ = torch.tensor(state_, dtype=torch.float).to(device)
            state = torch.tensor(state, dtype=torch.float).to(device)
            actions = torch.tensor(actions, dtype=torch.float).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)

            # Critics gradient step
            _, _, action_probs_ = self.actor.sample_normal(state_)
            
            with torch.no_grad():
                q1_target_value = self.critic_1_target.forward(state_)
                q2_target_value = self.critic_2_target.forward(state_)
                q_target_value = torch.min(q2_target_value, q1_target_value) * action_probs_
                q_hat = reward.view(batch_size, -1) + gamma * ~(dones.view(batch_size, -1)) * q_target_value
            q1_value = self.critic_1.forward(state).gather(1, actions.long())
            q2_value = self.critic_2.forward(state).gather(1, actions.long())
            q1_loss = 0.5 * F.mse_loss(q1_value, q_hat)
            q2_loss = 0.5 * F.mse_loss(q2_value, q_hat)
            
            q_loss = q1_loss + q2_loss
            self.critic_1.zero_grad()
            self.critic_2.zero_grad()
            q_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # Policy gradient step
            _, log_probs, action_probs = self.actor.sample_normal(state)
            q1_value = self.critic_1.forward(state)
            q2_value = self.critic_2.forward(state)
            q_value = torch.min(q1_value, q2_value)
            actor_loss = (action_probs * (self.alpha * log_probs - q_value)).mean()
            self.actor.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Alpha gradient step
            _, log_probs, action_probs = self.actor.sample_normal(state)
            alpha_loss = -(action_probs * self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            # Target critic weights update
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            return
    
    def save_model(self):
        model_scripted = torch.jit.script(self.actor)
        model_scripted.save("models/unity_test" + "_final.pth")

def multiagent(observation):
    agent.gradient_step()
    action = agent.choose_action(observation)
    observation_, reward, terminated, _, _ = env.step(action)
    env.close()
    return (observation, action, reward, observation_, terminated), agent

agent = Agent()
if __name__ == '__main__':
    pbar = tqdm(total=total_plays)

    best_score = -100000
    score_history = []

    global_step = 1
    for i in range(total_plays):
        observation, _ = env.reset()
        done = False
        score = 0
        iter_steps = 1
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            done = terminated
            score += reward
            agent.remember(observation, action, reward, observation_, terminated)
            if global_step % N == 0:
                with mp.Pool(processes=2) as pool:
                    results = pool.map(multiagent, [observation, observation])
                    index_max = min(range(len(results)), key=results[0][2])
                    agent = results[1][index_max]
            iter_steps += 1
            global_step += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()

        pbar.update()

    pbar.close() 