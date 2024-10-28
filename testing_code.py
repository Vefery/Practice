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
dim = 128
memory_length = 1
lstm_layers = 1
critic_lr = 3e-4
actor_lr = critic_lr / 3.0
policy_noise = 0.2
noise_clip = 0.5
gamma=0.99
tau=0.005
max_size = 1000000
batch_size = 4
total_plays = 1000
num_epochs = 1
N = 1

#unity_env = UnityEnvironment("D:\Practice\SentisInfrence\Build\SentisInfrence.exe", no_graphics=True,)
#env = UnityToGymWrapper(unity_env)

env = gym.make("LunarLander-v3", continuous=True)

max_action=env.action_space.high
obs_dim = env.observation_space.shape
n_actions=env.action_space.shape[-1]
print(obs_dim)
print(n_actions)

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.done_memory = np.zeros((self.mem_size), dtype=bool)
        self.hidden_cr1 = (torch.zeros((1, dim), dtype=torch.float).to(device), torch.zeros((1, dim), dtype=torch.float).to(device))
        self.hidden_cr2 = (torch.zeros((1, dim), dtype=torch.float).to(device), torch.zeros((1, dim), dtype=torch.float).to(device))
        self.hidden_actor = (torch.zeros((1, dim), dtype=torch.float).to(device), torch.zeros((1, dim), dtype=torch.float).to(device))

    def store_transition(self, state, action, reward, state_, dones, hidden_actor, hidden_critic1, hidden_critic2):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = dones
        if hidden_actor is not None and hidden_critic1 is not None and hidden_critic2 is not None:
            self.hidden_cr1 = (hidden_critic1[0].detach(), hidden_critic1[0].detach())
            self.hidden_cr2 = (hidden_critic2[0].detach(), hidden_critic2[0].detach())
            self.hidden_actor = (hidden_actor[0].detach(), hidden_actor[0].detach())

        self.mem_cntr += 1
    
    def sample_history_sequence(self, history_length):
        max_mem = min(self.mem_cntr, self.mem_size)
        if max_mem <= history_length:
            hist_part1 = torch.zeros(1, n_actions * memory_length)
            hist_part2 = torch.zeros(1, obs_dim[0] * memory_length)
        else:
            hist_states = np.zeros([1, history_length, obs_dim[0]])
            hist_actions = np.zeros([1, history_length, n_actions])

            id = max_mem - 1
            hist_start_id = id - history_length
            if hist_start_id < 0:
                hist_start_id = 0
            # If exist done before the last experience (not including the done in id), start from the index next to the done.
            if len(np.where(self.done_memory[hist_start_id:id] == 1)[0]) != 0:
                hist_start_id = hist_start_id + (np.where(self.done_memory[hist_start_id:id] == True)[0][-1]) + 1
            hist_seg_len = id - hist_start_id
            hist_states[0, :hist_seg_len, :] = self.state_memory[hist_start_id:id]
            hist_actions[0, :hist_seg_len, :] = self.action_memory[hist_start_id:id]

            hist_part1 = torch.tensor(hist_actions, dtype=torch.float).reshape(1, -1)
            hist_part2 = torch.tensor(hist_states, dtype=torch.float).reshape(1, -1)

        dictionary = dict(history_actions=hist_part1,
                          history_obs=hist_part2)
        return dictionary

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
            
            hist= np.hstack((hist_actions, hist_states))

        hist_part1 = torch.tensor(hist_actions, dtype=torch.float).reshape(batch_size, -1)
        hist_part2 = torch.tensor(hist_states, dtype=torch.float).reshape(batch_size, -1)

        if batch_size <= max_mem:
            dictionary = dict(states=self.state_memory[batch],
                        states_=self.new_state_memory[batch],
                        actions=self.action_memory[batch],
                        rewards=self.reward_memory[batch],
                        dones=self.done_memory[batch],
                        history_actions=hist_part1,
                        history_obs=hist_part2,
                        hidden_critic1=self.hidden_cr1,
                        hidden_critic2=self.hidden_cr2,
                        hidden_actor=self.hidden_actor)
        else:
            dictionary = dict(history_actions=hist_part1,
                              history_obs=hist_part2,
                              hidden_critic1=self.hidden_cr1,
                              hidden_critic2=self.hidden_cr2,
                              hidden_actor=self.hidden_actor)
        
        return dictionary
    
class HistoryNetwork(nn.Module):
    def __init__(self):
        super(HistoryNetwork, self).__init__()

        self.fc_layers = nn.Sequential(nn.Linear((obs_dim[-1] + n_actions) * memory_length, dim),
                                       nn.ReLU())
        self.lstm_layers = nn.LSTM(dim, dim, num_layers=lstm_layers, batch_first=True)
        
        self.to(device)

    def forward(self, history_actions, history_obs, hidden: tuple[torch.Tensor, torch.Tensor]):
        history = torch.cat((history_actions, history_obs), dim=-1)

        x = self.fc_layers(history)
        if hidden is not None:
            out, hidden_ = self.lstm_layers(x)
        else:
            out, hidden_ = self.lstm_layers(x)

        return out, hidden_

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

    def forward(self, state, action, history_actions, history_obs, hidden: tuple[torch.Tensor, torch.Tensor]):
        me, hidden_ = self.critic_me(history_actions, history_obs, hidden)
        cf = self.critic_cf(torch.cat([state, action], dim=1))

        pi = self.critic_pi(torch.cat([me, cf], dim=1))
        return pi, hidden_

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        self.actor_me = HistoryNetwork()
        self.actor_cf = nn.Sequential(
            nn.Linear(obs_dim[-1], dim),
            nn.ReLU())
        self.actor_pi = nn.Sequential(nn.Linear(2 * dim, dim),
                                      nn.ReLU())
        self.loc = nn.Sequential(nn.Linear(dim, n_actions),
                                    nn.Tanh())
        self.scale = nn.Sequential(nn.Linear(dim, n_actions),
                                    nn.Tanh())


        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        #self.warmup_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=actor_lr, last_epoch=1000)

        self.to(device)

    def forward(self, state, history_actions, history_obs, hidden: tuple[torch.Tensor, torch.Tensor]):
        me, hidden_ = self.actor_me(history_actions, history_obs, hidden)
        cf = self.actor_cf(state)
        pi = self.actor_pi(torch.cat([me, cf], dim=-1))

        loc = self.loc(pi)
        scale_log = self.scale(pi)
        scale_log = torch.clamp(scale_log, min=-20, max=2)

        return loc, scale_log, hidden_

    def sample_normal(self, state, history_actions, history_obs, hidden=None):
        loc, scale_log, hidden_ = self.forward(state, history_actions, history_obs, hidden)
        scale = scale_log.exp()
        dist = Normal(loc, scale)

        sample = dist.rsample()

        action = torch.tanh(sample)*torch.tensor(env.action_space.high).to(device)

        return action, hidden_
    
class Agent():
    def __init__(self):
        self.memory = ReplayBuffer(max_size, obs_dim, n_actions)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                
        self.actor = ActorNetwork()
        self.actor_target = ActorNetwork()
        self.critic_1 = CriticNetwork()
        self.critic_2 = CriticNetwork()
        self.critic_1_target = CriticNetwork()
        self.critic_2_target = CriticNetwork()

        #self.actor.apply(init_weights)
        #self.critic_1.apply(init_weights)
        #self.critic_2.apply(init_weights)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.actor_target.load_state_dict(self.actor_target.state_dict())

    def choose_action(self, observation):
        dic = self.memory.sample_history_sequence(memory_length)
        history_actions = dic["history_actions"].to(device)
        history_obs = dic["history_obs"].to(device)
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(device)
        actions, _ = self.actor.sample_normal(state, history_actions, history_obs)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, dones, hidden_critic1, hidden_critic2, hidden_actor):
        self.memory.store_transition(state, action, reward, new_state, dones, hidden_critic1, hidden_critic2, hidden_actor)

    def gradient_step(self):
        if self.memory.mem_cntr < batch_size:

            return None, None, None
        
        for _ in range(num_epochs):
            dct = self.memory.sample_buffer_history(batch_size, memory_length)

            history_actions = dct["history_actions"].detach().to(device)
            history_obs = dct["history_obs"].detach().to(device)
            hidden_critic1 = dct["hidden_critic1"]
            hidden_critic2 = dct["hidden_critic2"]
            hidden_actor = dct["hidden_actor"]
            reward = torch.tensor(dct["rewards"], dtype=torch.float).to(device)
            state_ = torch.tensor(dct["states_"], dtype=torch.float).to(device)
            state = torch.tensor(dct["states"], dtype=torch.float).to(device)
            actions = torch.tensor(dct["actions"], dtype=torch.float).to(device)
            dones = torch.tensor(dct["dones"], dtype=torch.bool).to(device)

            # Critics gradient step
            actions_, _ = self.actor_target.sample_normal(state_, history_actions, history_obs, hidden_actor)
            
            with torch.no_grad():
                noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip).to(device)
                actions_ = (actions_ + noise).clamp(torch.tensor(env.action_space.low, device=device), torch.tensor(env.action_space.high, device=device))
                q1_target_value, _ = self.critic_1_target.forward(state_, actions_, history_actions, history_obs, hidden_critic1)
                q2_target_value, _ = self.critic_2_target.forward(state_, actions_, history_actions, history_obs, hidden_critic2)
                q_target_value = torch.min(q2_target_value, q1_target_value)
                q_hat = reward.view(batch_size, -1) + gamma * ~(dones.view(batch_size, -1)) * q_target_value
            q1_value, hidden_critic1_ = self.critic_1.forward(state, actions, history_actions, history_obs, hidden_critic1)
            q2_value, hidden_critic2_ = self.critic_2.forward(state, actions, history_actions, history_obs, hidden_critic2)
            q1_loss = 0.5 * F.mse_loss(q1_value, q_hat)
            q2_loss = 0.5 * F.mse_loss(q2_value, q_hat)
            
            q_loss = q1_loss + q2_loss
            self.critic_1.zero_grad()
            self.critic_2.zero_grad()
            q_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # Policy gradient step
            actions_reparam, hidden_actor_ = self.actor.sample_normal(state, history_actions, history_obs, hidden_actor)
            q1_value, _ = self.critic_1.forward(state, actions_reparam, history_actions, history_obs, hidden_critic1)
            actor_loss = -q1_value.mean()
            self.actor.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Target critic weights update
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return hidden_actor_, hidden_critic1_, hidden_critic2_
    
    def save_model(self):
        model_scripted = torch.jit.script(self.actor)
        model_scripted.save("models/walker_test" + "_final.pth")

pbar = tqdm(total=total_plays)
pbar.reset()


agent = Agent()
best_score = -1000000
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
        if iter_steps % N == 0:
            hidden_actor, hidden_critic1, hidden_critic2 = agent.gradient_step()
        agent.remember(observation, action, reward, observation_, terminated, hidden_actor, hidden_critic1, hidden_critic2)
        iter_steps += 1
        global_step += 1
        observation = observation_

    pbar.update()

pbar.close() 