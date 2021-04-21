import pickle
import numpy as np
from collections import defaultdict

dataset = 'diginetica'

fre = defaultdict(int)
seq = pickle.load(open('./datasets/'+dataset + '/all_train_seq.txt', 'rb'))
nodes = set()
lens = []
for s in seq:
    lens.append(len(s))
    for node in s:
        nodes.add(node)
    # for l in range(2,min(6,len(s)+1)):
    #     for i in range(0,len(s)-l):
    #         j = i+l
    #         assert j<=len(s)
    #         fre[tuple(s[i:j])] += 1
    for d in range(1,4):
        for i in range(0,len(s)-d):
            j = i+d
            fre[tuple([s[i],s[j]])] += 1 

print(sorted(fre.items(),key=lambda item:item[1],reverse=True)[0:1000])
print(len(nodes),np.mean(np.array(lens)))
print(max(lens))
print(len(fre.keys()))


