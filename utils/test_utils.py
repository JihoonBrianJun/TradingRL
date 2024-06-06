import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .metric_utils import compute_predictor_metrics
from .trajectory_utils import process_state_window, compute_reward

def save_model(model, save_dir, train_config):
    torch.save(model.state_dict(), f'{save_dir}.pt')
    with open(f'{save_dir}.json', 'w') as f:
        json.dump(train_config, f)
    
def test_predictor(model, loss_function, dataloader, test_bs,
                   data_len, pred_len, value_threshold, strong_threshold,
                   device, save_dir, train_config, best_test_loss=None, best_test_score=None,
                   save_ckpt=True, load_ckpt=False):
    if load_ckpt:
        model.load_state_dict(torch.load(f'{save_dir}.pt'))

    model.eval()
    test_loss = 0
    metric_list = ["correct", "rec_correct", "rec_tgt", "strong_prec_correct", "strong_prec_tgt"]
    metric_dict = dict((metric, 0) for metric in metric_list)
    
    for idx, batch in tqdm(enumerate(dataloader)):
        src = batch['src'].to(torch.float32).to(device)
        tgt = batch['tgt'].to(torch.float32).to(device)
        
        for step in range(pred_len):
            if step == 0:
                out = model(src, tgt[:,:data_len,:])
            else:
                out = model(src, torch.cat((tgt[:,:data_len,:], out[:,-step:].unsqueeze(dim=2)),dim=1))

        label = tgt[:,1:,:].squeeze(dim=2)                    
        loss = loss_function(out[:,-1],label[:,-1])
        test_loss += loss.detach().cpu().item()

        metrics = compute_predictor_metrics(out[:,-1], label[:,-1], value_threshold, strong_threshold)
        for key in metric_dict.keys():
            metric_dict[key] += metrics[key]
        
        if idx == 0:
            print(f'Out: {out[:,-1]}\n Label: {label[:,-1]}')
    
    avg_test_loss = np.sqrt(test_loss / (idx+1))
    correct_rate = metric_dict["correct"] / (test_bs*(idx+1))
    recall = metric_dict["rec_correct"] / metric_dict["rec_tgt"]
    precision_strong = metric_dict["rec_tgt"] / metric_dict["strong_prec_tgt"]
    test_score = (correct_rate + recall + precision_strong) / 3
    
    print(f'Test Average Loss: {avg_test_loss}')
    print(f'Test Correct: {metric_dict["correct"]} out of {test_bs*(idx+1)}')
    print(f'Test Recall: {metric_dict["rec_correct"]} out of {metric_dict["rec_tgt"]}')
    print(f'Test Precision (Strong): {metric_dict["strong_prec_correct"]} out of {metric_dict["strong_prec_tgt"]}')

    if save_ckpt:
        if best_test_loss is None:
            save_model(model, save_dir, train_config)
        # elif avg_test_loss < best_test_loss:
        elif test_score > best_test_score or correct_rate >= train_config['stop_correct_threshold']:
            save_model(model, save_dir, train_config)
            
    return avg_test_loss, correct_rate, test_score


def test_ppo_agents(Actor, horizon, window, fee, state_list, bs, device, save_dir, train_config, best_avg_reward,
                    save_ckpt=True, load_ckpt=False):
    if load_ckpt:
        Actor.load_state_dict(torch.load(f'{save_dir}.pt'))
    state_windows, actor_outputs, _ = process_state_window(state_list, Actor, None, horizon, window, device, bs, process_critic=False)
    
    reward_sum_list = []
    long_only_return_list = []
    win = 0
    for state_idx in range(len(state_list)):
        action_list, reward_list = compute_reward(state_windows[state_idx], actor_outputs[state_idx], horizon, window, fee)
        reward_sum = sum(reward_list)
        reward_sum_list.append(reward_sum)
        long_only_return = state_list[state_idx][-1][3] - state_list[state_idx][window-1][3]
        long_only_return_list.append(long_only_return)
        if reward_sum > long_only_return:
            win += 1
        if state_idx == 0:
            state_close_prices = [np.round(minute_state[3], 4) for minute_state in state_list[state_idx]]
            df = pd.DataFrame(state_close_prices, columns=['prev_close_price'])
            df['action'] = pd.DataFrame([np.nan] * window + [np.round(action, 4) for action in action_list])
            df['reward'] = pd.DataFrame([np.nan] * window + [np.round(reward, 4) for reward in reward_list])
            position_list = []
            position = 0
            for action in action_list:
                position += action
                position_list.append(position)
            df['position'] = pd.DataFrame([np.nan] * window + [np.round(position, 4) for position in position_list])
            print(df.tail(60))
            print(f'reward sum: {sum(reward_list)}')

    avg_reward = sum(reward_sum_list) / len(state_list)
    avg_long_return = sum(long_only_return_list) / len(state_list)
    print(f'Average Reward: {avg_reward} vs Best Average Reward: {best_avg_reward}')
    print(f'Average Reward: {avg_reward} vs Average Long Only Return: {avg_long_return}')
    print(f'Win: {win} / {len(state_list)}')

    if save_ckpt and avg_reward > best_avg_reward:
        save_model(Actor, save_dir, train_config)
        
    return avg_reward