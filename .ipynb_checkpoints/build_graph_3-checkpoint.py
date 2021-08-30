import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall/Nowplaying')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

all_train_seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))

seq2idx = {tuple([0]):0}
seq2fre = defaultdict(int)

nodes = set()
for s in all_train_seq:
    for node in s:
        nodes.add(node)
        seq2idx[tuple([node])] = node
        seq2fre[tuple([node])] += 1

print("nodes num:", len(nodes))

cnt = len(nodes) + 1
for s in tqdm(all_train_seq):
    for i in range(0,len(s)-2):
        for j in range(i+1,min(i+4, len(s)-1)):
            for k in range(j+1, min(i+5, len(s))):
                seq = tuple(sorted([s[i],s[j],s[k]]))
                seq2fre[seq] += 1


adj_all = defaultdict(list)
for x in tqdm(seq2fre.keys()):
    if len(x) != 1:
        for k in range(len(x)):
            y = tuple([x[k]])
            adj_all[seq2idx[y]].append(x)


for k,v in tqdm(adj_all.items()):
    v = sorted(v, reverse=True, key=lambda x:seq2fre[x])
    v_t = []
    num = 0
    for i in range(len(v)):
        if num < sample_num: 
            v_t.append(v[i])
            num += 1

            if v[i] not in seq2idx.keys():
                seq2idx[v[i]] = cnt                 
                cnt += 1
        
        adj_all[k] = []
        for t in v_t:
            adj_all[k].append(seq2idx[t])
            
for x in tqdm(seq2idx.keys()):
    x_idx = seq2idx[x]
    if len(x) != 1:
        for k in range(len(x)):
            y = tuple([x[k]])
            adj_all[x_idx].append(seq2idx[y])
        
                        
print("nodes: ",len(seq2idx))

adj = [[] for _ in range(cnt)]               
lenght = []

for k,v in tqdm(adj_all.items()):
    adj[k] = v
    lenght.append(len(v))
    
print(np.mean(lenght))

pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(seq2fre, open('datasets/' + dataset + '/seq2fre_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(seq2idx, open('datasets/' + dataset + '/seq2idx_' + str(sample_num) + '.pkl', 'wb'))