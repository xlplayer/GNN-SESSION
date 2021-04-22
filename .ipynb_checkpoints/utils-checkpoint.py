import numpy as np
import torch
from torch.utils.data import Dataset
import config
import copy

def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len
    
def handle_adj(adj, sample_num):
    n_entity = len(adj)
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor == sample_num:
            sampled_indices = list(range(n_neighbor))
        elif n_neighbor > sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])

    return adj_entity

class Data(Dataset):
    def __init__(self, adj, seq2fre, seq2idx, data, train_len=None):
        self.adj = adj
        self.seq2fre = seq2fre
        self.seq2idx = seq2idx
        inputs, mask, max_len = handle_data(data[0], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]

        max_n_node = self.max_len
        
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        
        adj_entity = np.zeros([max_n_node, config.sample_num], dtype=np.int64)
        for i in range(max_n_node):
            neighbor = []
            assert len(neighbor) <= config.sample_num
            cnt = 0
            for j in range(i-70,i+70):
                if j < 0 or j == i or j >= len(items) or items[i] == 0 or items[j]==0:
                    continue
                x = tuple(sorted([items[i], items[j]]))
                if x not in self.seq2idx.keys():
                    cnt += 1
                else:
                    neighbor.append([self.seq2idx[x],self.seq2fre[x]])

            neighbor = [v[0] for v in sorted(neighbor, key = lambda x:x[1], reverse=True)][0:]
            len1 = len(self.adj[items[i]])
            len2 = len(neighbor)
            # if cnt:
            #     print(cnt,end=",")\
            neighbor = list(copy.deepcopy(self.adj[items[i]]))+neighbor
            n_neighbor = len(neighbor)
            if n_neighbor == 0:
                continue
            p = [4 for _ in range(len1)] + [1 for _ in range(len2)]
            p /= np.sum(p)
            if n_neighbor >= config.sample_num:
                adj_entity[i] = np.random.choice(neighbor, size=config.sample_num, replace=False, p=p)
            else:
                adj_entity[i] = np.random.choice(neighbor, size=config.sample_num, replace=True, p=p)  
            # adj_entity[i] = np.array(list(self.adj[items[i]])+list(neighbor)+[0 for _ in range(config.sample_num-n_neighbor)])

        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input), torch.tensor(adj_entity)]

    def __len__(self):
        return self.length