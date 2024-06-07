import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def process_state(episode_list, Actor, Critic, horizon, window, device, state_bs, process_critic=True):
    state_feature_dim = len(episode_list[0][0])
    states = np.concatenate([np.stack([episode[t-window:t] for t in range(window, horizon)], axis=0) for episode in episode_list], axis=0)
    state_loader = DataLoader(torch.tensor(states).to(torch.float32), batch_size=state_bs, shuffle=False)
    
    Actor.eval()
    if process_critic:
        Critic.eval()
    
    actor_probs, actor_outputs, critic_outputs = [], [], []
    for _, batch in tqdm(enumerate(state_loader)):
        batch = batch.to(device)
        actor_prob = Actor(batch).detach().cpu().numpy()
        actor_probs.append(np.max(actor_prob, axis=1))
        actor_outputs.append(np.argmax(actor_prob, axis=1))
        if process_critic:
            critic_outputs.append(Critic(batch).detach().cpu().numpy())
    actor_probs = np.concatenate(actor_probs, axis=0).reshape(len(episode_list), horizon-window)
    actor_outputs = np.concatenate(actor_outputs, axis=0).reshape(len(episode_list), horizon-window)
    if process_critic:
        critic_outputs = np.concatenate(critic_outputs, axis=0).reshape(len(episode_list), horizon-window)
    states = states.reshape(len(episode_list), horizon-window, window, state_feature_dim)
    
    return states, actor_probs, actor_outputs, critic_outputs


def compute_reward(state_list, actor_output_list, action_bins, horizon, window, fee):
    position, vwap = 0, 0
    action_list, reward_list = [], []
    for t in range(horizon-window):
        window_close_price = state_list[t][-1][3]
        action = np.clip(actor_output_list[t] * 4/(action_bins-1) - 1, -1-position, 1-position)
        reward = -fee * (100 + window_close_price) * np.abs(action)
        if action > 0:
            if position >= 0:
                vwap = (position * vwap + action * window_close_price) / (position + action + 1e-8)
            elif position >= -action:
                reward += (vwap - window_close_price) * (-position)
                vwap = (position + action) * window_close_price / (position + action + 1e-8)
            else:
                reward += (vwap - window_close_price) * action
        elif action < 0:
            if position <= 0:
                vwap = (position * vwap + action * window_close_price) / (position + action + 1e-8)
            elif position <= -action:
                reward += (window_close_price - vwap) * position
                vwap = (position + action) * window_close_price / (position + action + 1e-8)
            else:
                reward += (window_close_price - vwap) * (-action)               
        position += action
        action_list.append(action)
        reward_list.append(reward)
    reward_list[-1] = reward_list[-1] + (window_close_price - vwap) * position
    
    return action_list, reward_list
    

def compute_trajectory(episode_list, Actor, action_bins, Critic, horizon, window, fee, device, state_bs):
    states, actor_probs, actor_outputs, critic_outputs = process_state(episode_list, Actor, Critic, horizon, window, device, state_bs)
    
    trajectory_list = []
    for episode_idx in range(len(episode_list)):
        _, reward_list = compute_reward(states[episode_idx], actor_outputs[episode_idx], action_bins, horizon, window, fee)
        reward_to_go_list, advantage_list = [], []
        for t in range(horizon-window-1, -1, -1):
            reward = reward_list[t]
            if t == horizon-window-1:
                advantage_list.append(reward)
                reward_to_go_list.append(reward)
            else:
                instantaneous_advantage = reward + critic_outputs[episode_idx][t+1] - critic_outputs[episode_idx][t]
                advantage_list.append(instantaneous_advantage + advantage_list[-1])
                reward_to_go_list.append(reward + reward_to_go_list[-1])
        reward_to_go_list = reward_to_go_list[::-1]
        advantage_list = advantage_list[::-1]

        trajectory_list.append({"states": torch.tensor(states[episode_idx], dtype=torch.float32),
                                "actor_probs": torch.tensor(actor_probs[episode_idx], dtype=torch.float32),
                                "actions": torch.tensor(actor_outputs[episode_idx], dtype=torch.long),
                                "reward_to_go": torch.tensor(reward_to_go_list, dtype=torch.float32),
                                "advantages": torch.tensor(advantage_list, dtype=torch.float32)})
    
    return trajectory_list