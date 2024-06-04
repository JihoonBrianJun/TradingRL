import torch

def ppo_loss(actor_outputs, actions, advantages, epsilon):
    actor_update_rate = actor_outputs / (actions+1e-8)
    clipped_actor_update_rate = torch.clip(actor_update_rate, 1-epsilon, 1+epsilon)
    # print(f'actor_update_rate : {actor_update_rate}')
    # print(f'clipped_actor_update_rate: {clipped_actor_update_rate}')
    return -torch.mean(torch.min(actor_update_rate * advantages, clipped_actor_update_rate * advantages))