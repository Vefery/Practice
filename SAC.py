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
from torch.distributions.normal import Normal
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

device = (torch.device("cuda"))
dim = 256
critic_lr = 3e-4
actor_lr = critic_lr / 3.0
reparam_noise = 1e-6
gamma=0.99
tau=0.005
alpha_start = 1
max_size = 10000000
batch_size = 128
total_plays = 100000
warmup_steps = 1000
num_epochs = 1
N = 1

#unity_env = UnityEnvironment("path to .exe")
#env = UnityToGymWrapper(unity_env)

env = gym.make("InvertedDoublePendulum-v4")

max_action=env.action_space.high
obs_dim = env.observation_space.shape
n_actions=env.action_space.shape[-1]

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
            nn.Linear(obs_dim[-1] + n_actions, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        #self.warmup_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=critic_lr, last_epoch=1000)

        self.to(device)

    def forward(self, state, action):
        q = self.layers(torch.cat([state, action], dim=1))

        return q

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(*obs_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.loc = nn.Linear(dim, n_actions)
        self.scale = nn.Linear(dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        #self.warmup_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=actor_lr, last_epoch=1000)

        self.to(device)

    def forward(self, state):
        relu = self.layers(state)

        loc = self.loc(relu)
        scale_log = self.scale(relu)
        scale_log = torch.clamp(scale_log, min=-20, max=2)

        return loc, scale_log

    def sample_normal(self, state, reparameterize=True):
        loc, scale_log = self.forward(state)
        scale = scale_log.exp()
        dist = Normal(loc, scale)

        if reparameterize:
            sample = dist.rsample()
        else:
            sample = dist.sample()

        action = torch.tanh(sample)*torch.tensor(env.action_space.high).to(device)
        log_probs = dist.log_prob(sample)

        return action, log_probs
    
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
        #self.alpha_warmup_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.alpha_optimizer, T_max=critic_lr, last_epoch=1000)
        self.target_entropy = -n_actions

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, dones):
        self.memory.store_transition(state, action, reward, new_state, dones)

    def gradient_step(self):
        if self.memory.mem_cntr < batch_size:
            return 0, 0, 0, 0
        
        for _ in range(num_epochs):
            state, actions, reward, state_, dones = self.memory.sample_buffer(batch_size)

            reward = torch.tensor(reward, dtype=torch.float).to(device)
            state_ = torch.tensor(state_, dtype=torch.float).to(device)
            state = torch.tensor(state, dtype=torch.float).to(device)
            actions = torch.tensor(actions, dtype=torch.float).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)

            # Critics gradient step
            actions_, log_probs_ = self.actor.sample_normal(state_, reparameterize=False)
            
            with torch.no_grad():
                q1_target_value = self.critic_1_target.forward(state_, actions_)
                q2_target_value = self.critic_2_target.forward(state_, actions_)
                q_target_value = torch.min(q2_target_value, q1_target_value)
                q_hat = reward.view(batch_size, -1) + gamma * ~(dones.view(batch_size, -1)) * (q_target_value - self.alpha * log_probs_)
            q1_value = self.critic_1.forward(state, actions)
            q2_value = self.critic_2.forward(state, actions)
            q1_loss = 0.5 * F.mse_loss(q1_value, q_hat)
            q2_loss = 0.5 * F.mse_loss(q2_value, q_hat)
            
            q_loss = q1_loss + q2_loss
            self.critic_1.zero_grad()
            self.critic_2.zero_grad()
            q_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # Policy gradient step
            actions_reparam, log_prob_reparam = self.actor.sample_normal(state, reparameterize=True)
            q1_value = self.critic_1.forward(state, actions_reparam)
            q2_value = self.critic_2.forward(state, actions_reparam)
            q_value = torch.min(q1_value, q2_value)
            actor_loss = (self.alpha * log_prob_reparam - q_value).mean()
            self.actor.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Alpha gradient step
            _, log_probs = self.actor.sample_normal(state, reparameterize=False)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            # Target critic weights update
            target_critic1_params = self.critic_1_target.named_parameters()
            critic1_params = self.critic_1.named_parameters()
            target_critic1_state_dict = dict(target_critic1_params)
            critic1_state_dict = dict(critic1_params)

            for name in target_critic1_state_dict:
                target_critic1_state_dict[name] = tau * target_critic1_state_dict[name].clone() + (1 - tau) * critic1_state_dict[name].clone()

            self.critic_1_target.load_state_dict(target_critic1_state_dict)

            target_critic2_params = self.critic_2_target.named_parameters()
            critic2_params = self.critic_2.named_parameters()
            target_critic2_state_dict = dict(target_critic2_params)
            critic2_state_dict = dict(critic2_params)

            for name in target_critic2_state_dict:
                target_critic2_state_dict[name] = tau * target_critic2_state_dict[name].clone() + (1 - tau) * critic2_state_dict[name].clone()

            self.critic_2_target.load_state_dict(target_critic2_state_dict)

            return q1_loss, q2_loss, actor_loss, alpha_loss
    
    def save_model(self):
        model_scripted = torch.jit.script(self.actor)
        model_scripted.save("models/double_pendulum" + "_final.pth")

pbar = tqdm(total=total_plays)
pbar.reset()
writer = SummaryWriter("logs/double_pendulum" + str(datetime.now().day) + str(datetime.now().hour) + str(datetime.now().minute))

writer.add_text(
          "Hyperparameters",
          "|param|value|\n|-|-|\n%s" % ("\n".join(
               [f"|Critic lr|{critic_lr}|",
                f"|Actor lr|{actor_lr}|",
                f"|Layer dim|{dim}|",
                f"|Batch size|{batch_size}|",
                f"|Gamma|{gamma}|",
                f"|Tau|{tau}|",
                ]
          )),
          int(str(datetime.now().day) + str(datetime.now().hour) + str(datetime.now().minute)))

agent = Agent()
best_score = env.reward_range[0]
score_history = []

global_step = 0
for i in range(total_plays):
    observation, _ = env.reset()
    done = False
    score = 0
    iter_steps = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
        agent.remember(observation, action, reward, observation_, terminated)
        if iter_steps % N == 0:
            q1l, q2l, actorl, alphal = agent.gradient_step()
            writer.add_scalar("charts/q1_loss", q1l, global_step=global_step)
            writer.add_scalar("charts/q2_loss", q2l, global_step=global_step)
            writer.add_scalar("charts/actor_loss", actorl, global_step=global_step)
            writer.add_scalar("charts/alpha_loss", alphal, global_step=global_step)
        iter_steps += 1
        global_step += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    # agent.critic_1.warmup_scheduler.step()
    # agent.critic_2.warmup_scheduler.step()
    # agent.actor.warmup_scheduler.step()
    # agent.alpha_warmup_scheduler.step()

    if avg_score > best_score:
        best_score = avg_score
        agent.save_model()

    writer.add_scalar("charts/reward", avg_score, global_step=global_step)
    writer.add_scalar("charts/step_count", iter_steps, global_step=global_step)
    pbar.update()

pbar.close()