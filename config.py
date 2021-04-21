dataset = 'Tmall'
num_node = 43098
dim = 100
epoch = 100
activate = 'relu'
sample_num = 12
batch_size = 100
lr = 0.001
lr_dc = 0.2
lr_dc_step = 3
l2 = 1e-5
hop = 2
dropout_gcn = 0.2
dropout_local = 0
dropout_global = 0.5
alpha = 0.2
patience = 100

if dataset == "diginetica":
    num_node = 43098
    dropout_gcn = 0.2
    dropout_local = 0.0

elif dataset == "Nowplaying":
    num_node = 60417
    dropout_gcn = 0.0
    dropout_local = 0.0

elif dataset == "Tmall":
    num_node = 40728
    dropout_gcn = 0.6
    dropout_local = 0.5
