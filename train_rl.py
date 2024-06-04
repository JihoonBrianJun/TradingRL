import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser

from utils.preprocess import preprocess_state
from utils.train_utils import train_ppo_agents
from model.agent import ActorNetwork, CriticNetwork


def main(args):
    save_dir = f'{args.save_dir}_{args.window}min'
    if not os.path.exists(save_dir.split('/')[0]):
        os.makedirs(save_dir.split('/')[0])
    train_config = {"horizon": args.horizon,
                    "window": args.window,
                    "model_dim": args.model_dim,
                    "n_head": args.n_head,
                    "num_layers": args.num_layers,
                    "initial_lr": args.lr,
                    "gamma": args.gamma}

    state_list = preprocess_state(args.data_path, args.horizon, args.hop)
    train_state_list = state_list[:int(len(state_list)*args.train_ratio)]
    test_state_list = state_list[int(len(state_list)*args.train_ratio):]
    
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    Actor = ActorNetwork(model_dim=args.model_dim, 
                         n_head=args.n_head, 
                         num_layers=args.num_layers, 
                         src_feature_dim=len(state_list[0][0]), 
                         window=args.window).to(device)
    Critic = CriticNetwork(model_dim=args.model_dim,
                           n_head=args.n_head,
                           num_layers=args.num_layers,
                           src_feature_dim=len(state_list[0][0]),
                           window=args.window).to(device)

    model_names = ["Actor", "Critic"]
    for idx, model in enumerate([Actor, Critic]):
        num_param = 0
        for _, param in model.named_parameters():
            num_param += param.numel()
        print(f'{model_names[idx]} param size: {num_param}')

    actor_optimizer = optim.AdamW(Actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    actor_scheduler = StepLR(actor_optimizer, step_size=1, gamma=args.gamma)

    critic_optimizer = optim.AdamW(Critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    critic_scheduler = StepLR(critic_optimizer, step_size=1, gamma=args.gamma)
    critic_loss_func = nn.MSELoss()
    
    train_ppo_agents(Actor, actor_optimizer, actor_scheduler,
                     Critic, critic_optimizer, critic_scheduler, critic_loss_func,
                     args.step, args.sample_size, args.horizon, args.window, args.fee, args.epsilon, 
                     train_state_list, test_state_list, args.bs, device,
                     save_dir, train_config)
                    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default='ckpt/rl')
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--sample_size', type=int, default=512)
    parser.add_argument('--horizon', type=int, default=250)
    parser.add_argument('--hop', type=int, default=50)
    parser.add_argument('--window', type=int, default=150)
    parser.add_argument('--fee', type=float, default=0.0003)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    args = parser.parse_args()
    main(args)