import torch
import numpy as np
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from tqdm import tqdm
from .test_utils import test_predictor, test_ppo_agents
from .trajectory_utils import compute_trajectory
from .ppo_utils import ppo_loss

def train_predictor(model, optimizer, scheduler, loss_function, max_norm,
                    train_loader, test_loader, test_bs,
                    data_len, pred_len, value_threshold, strong_threshold,
                    epoch, device, save_dir, train_config):
    
    best_test_loss = np.inf
    best_test_score = 0
    for epoch in tqdm(range(epoch)):
        if epoch % 10 == 0 and epoch != 0:
            test_loss, correct_rate, test_score = test_predictor(model, loss_function, test_loader, test_bs,
                                                                 data_len, pred_len, value_threshold, strong_threshold,
                                                                 device, save_dir, train_config, best_test_loss, best_test_score)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
            if test_score > best_test_score:
                best_test_score = test_score

        model.train()
        epoch_loss = 0
        for idx, batch in tqdm(enumerate(train_loader)):
            src = batch['src'].to(torch.float32).to(device)
            tgt = batch['tgt'].to(torch.float32).to(device)
            
            for step in range(pred_len):
                out = model(src, tgt[:,:data_len+step,:])
                label = tgt[:,1:data_len+step+1,:].squeeze(dim=2)
                loss = loss_function(out,label)
                loss.backward()

                clip_grad_value_(model.parameters(), clip_value=max_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.detach().cpu().item()     
        
        epoch_avg_loss = np.sqrt(epoch_loss/(idx+1))
        print(f'Epoch {epoch} Average Loss: {epoch_avg_loss}')
        scheduler.step()
        
        if epoch >= 10:
            if epoch_avg_loss < best_test_loss * train_config['stop_loss_ratio'] or correct_rate >= train_config['stop_correct_threshold']:
                print(f"Train early stop at epoch {epoch} (epoch_loss={epoch_avg_loss}, best_val_loss={best_test_loss}), correct_rate={correct_rate}")
                break
    
    test_predictor(model, loss_function, test_loader, test_bs,
                   data_len, pred_len, value_threshold, strong_threshold,
                   device, save_dir, train_config, save_ckpt=False, load_ckpt=True)



def train_ppo_agents(Actor, action_bins, actor_optimizer, actor_scheduler,
                     Critic, critic_optimizer, critic_scheduler, critic_loss_func,
                     epoch, step_per_epoch, sample_size, horizon, window, fee, epsilon, 
                     train_episode_list, test_episode_list, bs, device,
                     save_dir, train_config):
    
    bs = min(bs, sample_size)
    best_avg_reward = -np.inf
    nan_stop = False
    for epoch_idx in range(epoch):
        if epoch_idx % 10 == 0 and epoch_idx != 0:
            avg_reward = test_ppo_agents(Actor, action_bins, horizon, window, fee, test_episode_list, bs, device, save_dir, train_config, best_avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            
        sample_idx = np.random.choice(np.arange(len(train_episode_list)), size=sample_size, replace=False)
        trajectory_list = compute_trajectory(train_episode_list[sample_idx], Actor, action_bins, Critic, horizon, window, fee, device, bs*(horizon-window))
        train_loader = DataLoader(trajectory_list, batch_size=bs, shuffle=True)
        
        Actor.train()
        Critic.train()
        actor_step_loss, critic_step_loss = 0, 0
        for _ in range(step_per_epoch):
            for idx, batch in tqdm(enumerate(train_loader)):
                batch_states = batch["states"].reshape(bs*(horizon-window), window, -1).to(device)
                batch_old_actor_probs = batch["actor_probs"].reshape(-1).to(device)
                batch_old_actions = batch["actions"].reshape(-1).to(device)
                batch_advantages = batch["advantages"].reshape(-1).to(device)
                batch_reward_to_go = batch["reward_to_go"].reshape(-1).to(device)
            
                batch_actor_outputs = Actor(batch_states)
                actor_loss = ppo_loss(batch_actor_outputs, batch_old_actor_probs, batch_old_actions, batch_advantages, epsilon, bs*(horizon-window))
                actor_loss.backward()
                actor_optimizer.step()
                actor_optimizer.zero_grad()
                actor_scheduler.step()
                actor_step_loss += actor_loss.detach().cpu().item()
                
                batch_critic_outputs = Critic(batch_states)
                critic_loss = critic_loss_func(batch_critic_outputs, batch_reward_to_go)
                critic_loss.backward()
                critic_optimizer.step()
                critic_optimizer.zero_grad()
                critic_scheduler.step()
                critic_step_loss += critic_loss.detach().cpu().item()
                
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    nan_stop = True
                    break
            if nan_stop:
                break
                # print(f'batch_actions: {batch_actions}')
                # print(f'batch_advantages: {batch_advantages}')
                # print(f'batch_reward_to_go: {batch_reward_to_go}')
                # print(f'batch_actor_outputs: {batch_actor_outputs}')
                # print(f'batch_critic_outputs: {batch_critic_outputs}')
        if nan_stop:
            print(f"nan_stop at Epoch {epoch_idx}")
            break
        
        actor_step_avg_loss = actor_step_loss/(idx+1)
        critic_step_avg_loss = critic_step_loss/(idx+1)
        print(f'Epoch {epoch_idx} Actor Average Loss: {actor_step_avg_loss}')
        print(f'Epoch {epoch_idx} Critic Average Loss: {critic_step_avg_loss}')
    
    test_ppo_agents(Actor, action_bins, horizon, window, fee, test_episode_list, bs*(horizon-window), device, save_dir, train_config, best_avg_reward,
                    save_ckpt=False, load_ckpt=True)