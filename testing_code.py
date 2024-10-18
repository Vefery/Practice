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
memory_length = 510
lstm_layers = 1
critic_lr = 3e-4
actor_lr = critic_lr / 3.0
reparam_noise = 1e-6
gamma=0.99
tau=0.005
alpha_start = 1
max_size = 1000000
batch_size = 512
total_plays = 1000
num_epochs = 3
N = 1

unity_env = UnityEnvironment("D:\Practice\AIProject\Build\AIProject.exe", no_graphics=True,)
env = UnityToGymWrapper(unity_env)

#env = gym.make("MountainCarContinuous-v0")

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
    
    def sample_buffer_history(self, batch_size, history_length):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.randint(history_length, max_mem, batch_size)

        if history_length == 0:
                hist_states = np.zeros([batch_size, 1, obs_dim[0]])
                hist_actions = np.zeros([batch_size, 1, n_actions])
                hist_states_len = np.zeros(batch_size)
        else:
            hist_states = np.zeros([batch_size, history_length, obs_dim[0]])
            hist_actions = np.zeros([batch_size, history_length, n_actions])
            hist_states_len = history_length * np.ones(batch_size)

            for i, id, in enumerate(batch):
                hist_start_id = id - history_length
                if hist_start_id < 0:
                    hist_start_id = 0
                # If exist done before the last experience (not including the done in id), start from the index next to the done.
                if len(np.where(self.done_memory[hist_start_id:id] == 1)[0]) != 0:
                    hist_start_id = hist_start_id + (np.where(self.done_memory[hist_start_id:id] == True)[0][-1]) + 1
                hist_seg_len = id - hist_start_id
                hist_states_len[i] = hist_seg_len
                hist_states[i, :hist_seg_len, :] = self.state_memory[hist_start_id:id]
                hist_actions[i, :hist_seg_len, :] = self.action_memory[hist_start_id:id]

        hist_part1 = torch.tensor(hist_actions, dtype=torch.float).reshape(batch_size, -1)
        hist_part2 = torch.tensor(hist_states, dtype=torch.float).reshape(batch_size, -1)

        if batch_size <= max_mem:
            dictionary = dict(states=self.state_memory[batch],
                        states_=self.new_state_memory[batch],
                        actions=self.action_memory[batch],
                        rewards=self.reward_memory[batch],
                        dones=self.done_memory[batch],
                        history_actions=hist_part1,
                        histpry_obs=hist_part2)
        else:
            dictionary = dict(history_actions=hist_part1, histpry_obs=hist_part2)
        
        return dictionary
    
class HistoryNetwork(nn.Module):
    def __init__(self):
        super(HistoryNetwork, self).__init__()

        self.fc_layers = nn.Sequential(nn.Linear((obs_dim[-1] + n_actions) * memory_length, dim),
                                       nn.ReLU())
        self.lstm_layers = nn.LSTM(dim, dim, num_layers=lstm_layers, batch_first=True)
        
        self.to(device)

    def forward(self, history_actions, history_obs):
        history = torch.cat((history_actions, history_obs), dim=-1)
        x = self.fc_layers(history)
        out, _ = self.lstm_layers(x)

        return out

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.critic_me = HistoryNetwork()

        self.critic_cf = nn.Sequential(
            nn.Linear(obs_dim[-1] + n_actions, dim),
            nn.ReLU())
        
        self.critic_pi = nn.Sequential(nn.Linear(dim * 2, dim),
                                       nn.ReLU(),
                                       nn.Linear(dim, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        #self.warmup_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=critic_lr, last_epoch=1000)

        self.to(device)

    def forward(self, state, action, history_actions, history_obs):
        me = self.critic_me(history_actions, history_obs)
        cf = self.critic_cf(torch.cat([state, action], dim=1))

        pi = self.critic_pi(torch.cat([me, cf], dim=1))
        return pi

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        self.actor_me = HistoryNetwork()
        self.actor_cf = nn.Sequential(
            nn.Linear(obs_dim[-1], dim),
            nn.ReLU())
        self.actor_pi = nn.Sequential(nn.Linear(2 * dim, dim),
                                      nn.ReLU())
        self.loc = nn.Linear(dim, n_actions)
        self.scale = nn.Linear(dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        #self.warmup_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=actor_lr, last_epoch=1000)

        self.to(device)

    def forward(self, state, history_actions, history_obs):
        me = self.actor_me(history_actions, history_obs)
        cf = self.actor_cf(state)
        pi = self.actor_pi(torch.cat([me, cf], dim=-1))

        loc = self.loc(pi)
        scale_log = self.scale(pi)
        scale_log = torch.clamp(scale_log, min=-20, max=2)

        return loc, scale_log

    def sample_normal(self, state, history_actions, history_obs):
        loc, scale_log = self.forward(state, history_actions, history_obs)
        scale = scale_log.exp()
        dist = Normal(loc, scale)


        sample = dist.rsample()

        action = torch.tanh(sample)*torch.tensor(env.action_space.high).to(device)
        log_probs = dist.log_prob(sample)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    
class Agent():
    def __init__(self):
        self.memory = ReplayBuffer(max_size, obs_dim, n_actions)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                
        self.actor = ActorNetwork()
        self.critic_1 = CriticNetwork()
        self.critic_2 = CriticNetwork()
        self.critic_1_target = CriticNetwork()
        self.critic_2_target = CriticNetwork()

        self.actor.apply(init_weights)
        self.critic_1.apply(init_weights)
        self.critic_2.apply(init_weights)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.alpha = alpha_start
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_lr)
        self.target_entropy = -n_actions

    def choose_action(self, observation, history_actions=None, history_obs=None):
        if history_actions is None and history_obs is None:
            history_actions = torch.zeros(1, n_actions * memory_length).to(device)
            history_obs = torch.zeros(1, obs_dim[0] * memory_length).to(device)
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(device)
        actions, _ = self.actor.sample_normal(state, history_actions, history_obs)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, dones):
        self.memory.store_transition(state, action, reward, new_state, dones)

    def gradient_step(self):
        if self.memory.mem_cntr < batch_size:
            return
        
        for _ in range(num_epochs):
            dct = self.memory.sample_buffer_history(batch_size, memory_length)

            history_actions = dct["history_actions"].detach().to(device)
            history_obs = dct["history_obs"].detach().to(device)
            reward = torch.tensor(dct["rewards"], dtype=torch.float).to(device)
            state_ = torch.tensor(dct["states_"], dtype=torch.float).to(device)
            state = torch.tensor(dct["states"], dtype=torch.float).to(device)
            actions = torch.tensor(dct["actions"], dtype=torch.float).to(device)
            dones = torch.tensor(dct["dones"], dtype=torch.bool).to(device)

            # Critics gradient step
            actions_, log_probs_ = self.actor.sample_normal(state_, history_actions, history_obs)
            
            with torch.no_grad():
                q1_target_value = self.critic_1_target.forward(state_, actions_, history_actions, history_obs)
                q2_target_value = self.critic_2_target.forward(state_, actions_, history_actions, history_obs)
                q_target_value = torch.min(q2_target_value, q1_target_value)
                q_hat = reward.view(batch_size, -1) + gamma * ~(dones.view(batch_size, -1)) * (q_target_value - self.alpha * log_probs_)
            q1_value = self.critic_1.forward(state, actions, history_actions, history_obs)
            q2_value = self.critic_2.forward(state, actions, history_actions, history_obs)
            q1_loss = 0.5 * F.mse_loss(q1_value, q_hat)
            q2_loss = 0.5 * F.mse_loss(q2_value, q_hat)
            
            q_loss = q1_loss + q2_loss
            self.critic_1.zero_grad()
            self.critic_2.zero_grad()
            q_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # Policy gradient step
            actions_reparam, log_prob_reparam = self.actor.sample_normal(state, history_actions, history_obs)
            q1_value = self.critic_1.forward(state, actions_reparam, history_actions, history_obs)
            q2_value = self.critic_2.forward(state, actions_reparam, history_actions, history_obs)
            q_value = torch.min(q1_value, q2_value)
            actor_loss = (self.alpha * log_prob_reparam - q_value).mean()
            self.actor.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Alpha gradient step
            alpha_loss = -(self.log_alpha * (log_prob_reparam + self.target_entropy).detach()).mean()
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
    
    def save_model(self):
        model_scripted = torch.jit.script(self.actor)
        model_scripted.save("models/unity_test" + "_final.pth")

pbar = tqdm(total=total_plays)
pbar.reset()

agent = Agent()
best_score = env.reward_range[0]
score_history = []

global_step = 0
for i in range(total_plays):
    observation = env.reset()
    done = False
    score = 0
    iter_steps = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, terminated, _ = env.step(action)
        done = terminated
        if terminated and iter_steps > 748:
            terminated = False
        score += reward
        agent.remember(observation, action, reward, observation_, terminated)
        if iter_steps % N == 0:
            agent.gradient_step()
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