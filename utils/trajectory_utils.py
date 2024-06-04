import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def process_state_window(state_list, Actor, Critic, horizon, window, device, state_bs, process_critic=True):
    state_feature_dim = len(state_list[0][0])
    state_windows = np.concatenate([np.stack([state[t-window:t] for t in range(window, horizon)], axis=0) for state in state_list], axis=0)
    state_loader = DataLoader(torch.tensor(state_windows).to(torch.float32), batch_size=state_bs, shuffle=False)
    
    Actor.eval()
    if process_critic:
        Critic.eval()
    
    actor_outputs, critic_outputs = [], []
    for _, batch in tqdm(enumerate(state_loader)):
        batch = batch.to(device)
        actor_outputs.append(Actor(batch).detach().cpu().numpy())
        if process_critic:
            critic_outputs.append(Critic(batch).detach().cpu().numpy())
    actor_outputs = np.concatenate(actor_outputs, axis=0).reshape(len(state_list), horizon-window)
    if process_critic:
        critic_outputs = np.concatenate(critic_outputs, axis=0).reshape(len(state_list), horizon-window)
    state_windows = state_windows.reshape(len(state_list), horizon-window, window, state_feature_dim)
    
    return state_windows, actor_outputs, critic_outputs


def compute_reward(state_window_list, actor_output_list, horizon, window, fee):
    position, vwap = 0, 0
    action_list, reward_list = [], []
    for t in range(horizon-window):
        state_window = state_window_list[t]
        window_close_price = state_window[-1][3]
        action = np.clip(actor_output_list[t], -1-position, 1-position)
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
    

def compute_trajectory(state_list, Actor, Critic, horizon, window, fee, device, state_bs):
    state_windows, actor_outputs, critic_outputs = process_state_window(state_list, Actor, Critic, horizon, window, device, state_bs)
    
    trajectory_list = []
    for state_idx in range(len(state_list)):
        action_list, reward_list = compute_reward(state_windows[state_idx], actor_outputs[state_idx], horizon, window, fee)
        reward_to_go_list, advantage_list = [], []
        for t in range(horizon-window-2, -1, -1):
            reward = reward_list[t]
            instantaneous_advantage = reward + critic_outputs[state_idx][t+1] - critic_outputs[state_idx][t]
            if t == horizon-window-2:
                advantage_list.append(instantaneous_advantage)
                reward_to_go_list.append(reward)
            else:            
                advantage_list.append(instantaneous_advantage + advantage_list[-1])
                reward_to_go_list.append(reward + reward_to_go_list[-1])
        reward_to_go_list = reward_to_go_list[::-1]
        advantage_list = advantage_list[::-1]

        trajectory_list.append({"states": torch.tensor(state_windows[state_idx][:-1], dtype=torch.float32),
                                "actions": torch.tensor(action_list[:-1], dtype=torch.float32),
                                "reward_to_go": torch.tensor(reward_to_go_list, dtype=torch.float32),
                                "advantages": torch.tensor(advantage_list, dtype=torch.float32)})
    
    return trajectory_list