import torch

def ppo_loss(actor_outputs, old_actor_probs, old_actions, advantages, epsilon, bs):
    actor_probs = actor_outputs[torch.arange(bs), old_actions]
    actor_update_rate = actor_probs / (old_actor_probs + 1e-8)
    clipped_actor_update_rate = torch.clip(actor_update_rate, 1-epsilon, 1+epsilon)
    # print(f'actor_update_rate : {actor_update_rate}')
    # print(f'clipped_actor_update_rate: {clipped_actor_update_rate}')
    return -torch.mean(torch.min(actor_update_rate * advantages, clipped_actor_update_rate * advantages))