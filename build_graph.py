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
    for i in range(0,len(s)-1):
        for j in range(i+1,min(i+5, len(s))):
            seq = tuple(sorted([s[i],s[j]]))
            if seq not in seq2idx.keys():
                seq2fre[seq] = 1
                seq2idx[seq] = cnt
                cnt += 1
            else:
                seq2fre[seq] += 1


# for s in tqdm(all_train_seq):
#     for i in range(0,len(s)-2):
#         for j in range(i+1,min(i+2, len(s)-1)):
#             for k in range(j+1, min(i+3, len(s))):
#                 seq = tuple(sorted([s[i],s[j],s[k]]))
#                 if seq not in seq2idx.keys():
#                     seq2fre[seq] = 1
#                     seq2idx[seq] = cnt
#                     cnt += 1
#                 else:
#                     seq2fre[seq] += 1

print("nodes:",len(seq2fre))

adj1 = [defaultdict(int) for _ in range(cnt)]
adj = [[] for _ in range(cnt)]

for x in tqdm(seq2idx.keys()):
    x_idx = seq2idx[x]
    if len(x) != 1:
        for k in range(len(x)):
            y = tuple([x[k]])
            y_idx = seq2idx[y]
            adj1[x_idx][y_idx] = seq2fre[y]
            adj1[y_idx][x_idx] = seq2fre[x]
                
lenght = []
for t in tqdm(range(cnt)):
    adj[t] = [v[0] for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])][0:sample_num]
    lenght.append(len(adj[t]))
print(np.mean(lenght))

pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(seq2fre, open('datasets/' + dataset + '/seq2fre_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(seq2idx, open('datasets/' + dataset + '/seq2idx_' + str(sample_num) + '.pkl', 'wb'))