import json
import torch
import numpy as np
import pandas as pd
from .trajectory_utils import process_state, compute_reward

def save_model(model, save_dir, train_config):
    torch.save(model.state_dict(), f'{save_dir}.pt')
    with open(f'{save_dir}.json', 'w') as f:
        json.dump(train_config, f)
    

def test_ppo_agents(Actor, action_bins, horizon, window, fee, episode_list, bs, device, save_dir, train_config, best_avg_reward,
                    save_ckpt=True, load_ckpt=False):
    if load_ckpt:
        Actor.load_state_dict(torch.load(f'{save_dir}.pt'))
    states, _, actor_outputs, _ = process_state(episode_list, Actor, None, horizon, window, device, bs, process_critic=False)
    
    reward_sum_list = []
    long_only_return_list = []
    win = 0
    for episode_idx in range(len(episode_list)):
        action_list, reward_list = compute_reward(states[episode_idx], actor_outputs[episode_idx], action_bins, horizon, window, fee)
        reward_sum = sum(reward_list)
        reward_sum_list.append(reward_sum)
        long_only_return = episode_list[episode_idx][-1][3] - episode_list[episode_idx][window-1][3]
        long_only_return -= fee * ((100 + episode_list[episode_idx][-1][3]) + (100 + episode_list[episode_idx][window-1][3]))
        long_only_return_list.append(long_only_return)
        if reward_sum > long_only_return:
            win += 1
        if episode_idx == 0:
            state_close_prices = [np.round(minute_state[3], 4) for minute_state in episode_list[episode_idx]]
            df = pd.DataFrame(state_close_prices, columns=['prev_close_price'])
            df['action'] = pd.DataFrame([np.nan] * window + [np.round(action, 4) for action in action_list])
            df['reward'] = pd.DataFrame([np.nan] * window + [np.round(reward, 4) for reward in reward_list])
            position_list = []
            position = 0
            for action in action_list:
                position += action
                position_list.append(position)
            df['position'] = pd.DataFrame([np.nan] * window + [np.round(position, 4) for position in position_list])
            print(df)
            print(f'reward sum: {sum(reward_list)}')

    avg_reward = sum(reward_sum_list) / len(episode_list)
    avg_long_return = sum(long_only_return_list) / len(episode_list)
    print(f'Average Reward: {avg_reward} vs Best Average Reward: {best_avg_reward}')
    print(f'Average Reward: {avg_reward} vs Average Long Only Return: {avg_long_return}')
    print(f'Win: {win} / {len(episode_list)}')

    if save_ckpt and avg_reward > best_avg_reward:
        save_model(Actor, save_dir, train_config)
        
    return avg_reward