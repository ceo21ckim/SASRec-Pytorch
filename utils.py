import torch, copy, random, sys, os, wandb, tqdm

import numpy as np
from torch.utils.data import Dataset, DataLoader 
from collections import defaultdict

def random_sample(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    
    return t 

def data_partition(fname):
    num_users = 0 
    num_items = 0 
    users = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        num_users = max(u, num_users)
        num_items = max(i, num_items)
        users[u].append(i)

    for user in users:
        nfeedback = len(users[user])
        if nfeedback < 3:
            user_train[user] = users[user]
            user_valid[user] = []
            user_test[user] = []

        else:
            user_train[user] = users[user][:-2]
            user_valid[user] = []
            user_valid[user].append(users[user][-2])
            user_test[user] = []
            user_test[user].append(users[user][-1])

    return [user_train, user_valid, user_test, num_users, num_items]


def evaluate(args, model, dataset):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.max_length], dtype=np.int32)
        idx = args.max_length - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        u = torch.tensor([u], device=args.device)
        seq = torch.tensor([seq], device=args.device)
        item_idx = torch.tensor(item_idx, device=args.device)
        predictions = -model.predict(u, seq, item_idx)
        predictions = predictions[0] # - for 1st argsort DESC
    

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.K:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(args, model, dataset):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.max_length], dtype=np.int32)
        idx = args.max_length - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        u = torch.tensor([u], device=args.device)
        seq = torch.tensor([seq], device=args.device)
        item_idx = torch.tensor(item_idx, device=args.device)
        predictions = -model.predict(u, seq, item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < args.K:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


class SequenceDataset(Dataset):
    def __init__(self, args, dataset):
        self.args = args 
        self.users = list(dataset.keys())
        self.items = list(dataset.values())

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        self.user = self.users[idx]
        self.pos = self.items[idx]
        return {
           'users': torch.tensor(self.user),
           'items': torch.tensor(self.pos), 
           'args': self.args
        }


def collate_fn(samples):
    users = [sample['users'] for sample in samples]
    items = [sample['items'] for sample in samples]
    args = samples[0]['args']
    max_length = args.max_length 
    num_items = args.num_items 

    seqs = []
    poses = []
    negs = []


    for user, item in zip(users, items):
        seq = np.zeros([max_length], dtype=np.int32)
        pos = np.zeros([max_length], dtype=np.int32)
        neg = np.zeros([max_length], dtype=np.int32)
        if not item.tolist():
            break 
        nxt = item[-1]
        idx = max_length - 1

        ts = set(item)
        for i in reversed(item[:-1]):
            seq[idx] = i 
            pos[idx] = nxt 
            if nxt != 0:
                neg[idx] = random_sample(1, num_items + 1, ts)
            
            nxt = i 
            idx -= 1
            if idx == -1 :
                break

        seqs.append(seq)
        poses.append(pos)
        negs.append(neg)

    return (
        torch.tensor(users), 
        torch.tensor(seqs), 
        torch.tensor(poses), 
        torch.tensor(negs)
    )

def get_loader(args, d_set, shuffle=True, n_workers=0):
    dataset = SequenceDataset(args, d_set)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=n_workers)


def train(args, model, train_loader, dataset, optimizer, criterion):
    best_metrics = float('-inf')
    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm.tqdm(train_loader, desc='training...'):
            user = batch[0].to(args.device)
            seq = batch[1].to(args.device)
            pos = batch[2].to(args.device)
            neg = batch[3].to(args.device)
            
            pos_logits, neg_logits = model(user, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            optimizer.zero_grad()
            pos = pos.detach().cpu()
            indices = np.where(pos.detach().cpu() != 0)
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        print(f'[{epoch + 1}/{args.num_epochs}]')
        print(f'loss: {loss.item():.4f}')

        if args.use_wandb:
            wandb.log({'train_loss': loss})

        if (epoch + 1) % 1 == 0 :
            model.eval()
            print('Evaluating..')

            t_test = evaluate(args, model, dataset)
            t_valid = evaluate_valid(args, model, dataset)

            print(f'\nEpoch: [{epoch + 1}/{args.num_epochs}]\nvalid: (NDCG@{args.K}: {t_valid[0]:.4f}, HR@{args.K}: {t_valid[1]:.4f}\t test: (NDCG@{args.K}: {t_test[0]:.4f}, HR@{args.K}: {t_test[1]:.4f}))')
            if args.use_wandb:
                wandb.log({f'valid_NDCG@{args.K}':t_valid[0], f'valid_HR@{args.K}': t_valid[1], f'test_NDCG@{args.K}': t_test[0], f'test_HR@{args.K}': t_test[1]})

            if best_metrics < t_valid[0]:
                best_metrics = t_valid[0]

                if not os.path.exists('model_parameters'):
                    os.makedirs('model_parameters')
                    torch.save(model.state_dict(), os.path.join('model_parameters', f'{model._get_name()}-hidden={args.hidden_dim}-lr={args.lr}-epoch={args.num_epochs}-maxlen={args.max_length}.pt'))
            