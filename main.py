import os
import time
import torch
from torch import nn, optim 
import argparse

from models import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--K', default=10, type=int)
# parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--max_length', default=50, type=int)
parser.add_argument('--hidden_dim', default=50, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dr_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--use_wandb', default=False, type=str2bool)


args = parser.parse_args()

if __name__ == '__main__':

    if args.use_wandb:
        wandb.init(project='Sequential', 
                config = {
                    'num_epochs': args.num_epochs, 
                    'max_length': args.max_length, 
                    'num_blocks': args.num_layers, 
                    'batch_size': args.batch_size, 
                    'learning_rate': args.lr, 
                    'dr_rate': args.dr_rate
                })

    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, num_users, num_items] = dataset 
    
    args.num_users = num_users 
    args.num_items = num_items 

    model = SASRec(args, num_users, num_items).to(args.device)

    model.train()

    if args.state_dict_path is not None:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(args, model, dataset)
        print(f'test NDCG@{args.K}: {t_test[0]:.4f}\tHR@{args.K}: {t_test[0]:.4f}')
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.98))

    train_loader = get_loader(args, user_train, shuffle=True, n_workers=0)

    train(args, model, train_loader, dataset, optimizer, criterion)