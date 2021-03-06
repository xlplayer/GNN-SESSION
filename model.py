import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import math
from aggregator import LocalAggregator, GlobalAggregator

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    # if is_cuda:
    #     U = U.cuda()
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)



def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

class SessionGraph(nn.Module):
    def __init__(self, adj):
        super(SessionGraph, self).__init__()
        self.adj_all = adj
        self.num_node = len(adj)

        self.embedding = nn.Embedding(self.num_node, config.dim)
        self.pos_embedding = nn.Embedding(200, config.dim)

        # Aggregator
        self.local_agg = LocalAggregator(config.dim, config.alpha)
        self.global_agg = []
        for i in range(config.hop):
            if config.activate == 'relu':
                agg = GlobalAggregator(config.dim, config.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(config.dim, config.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        
        # Parameters
        self.w_s = nn.Parameter(torch.Tensor(config.dim, 1))
        self.w_g = nn.Parameter(torch.Tensor(config.dim, 1))
        self.w_x = nn.Parameter(torch.Tensor(config.dim, config.dim))

        self.w_1 = nn.Parameter(torch.Tensor(2 * config.dim, config.dim))
        self.w_2 = nn.Parameter(torch.Tensor(config.dim, 1))
        self.glu1 = nn.Linear(config.dim, config.dim)
        self.glu2 = nn.Linear(config.dim, config.dim, bias=False)
        self.linear_transform = nn.Linear(config.dim, config.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(config.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target):
        return self.adj_all[target.view(-1)]

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        b = self.embedding.weight[1:config.num_node]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def forward(self, inputs, adj, mask_item, item, first_adj):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj)

        # global
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1) #b*s*h
        
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2) #b*1*h
        
        # first_adj_unsqueeze = torch.unsqueeze(first_adj, -1) #b*s*l*1
        # first_adj_pad = torch.cat([torch.zeros_like(first_adj_unsqueeze), first_adj_unsqueeze], dim=-1) #b*s*l*2
        # first_adj_emb = self.embedding(first_adj_pad) #b*s*l*2*h
        # first_adj_emb = torch.matmul(first_adj_emb * sum_item_emb.unsqueeze(1).unsqueeze(1).repeat(1,first_adj_emb.shape[1],first_adj_emb.shape[2],first_adj_emb.shape[3],1), self.w_x)
        # h_unsqueeze = h.unsqueeze(-1).unsqueeze(2) #b*s*1*h*1
        # s = torch.matmul(first_adj_emb,h_unsqueeze) #b*s*l*2*1
        # y_hard = gumbel_softmax(s.squeeze(-1), hard=True) #b*s*l*2
        # first_adj = torch.sum(first_adj_pad * y_hard, dim=-1) #b*s*l
        # first_adj = first_adj.long()

        neighbor_num = [config.sample_num, 2]
        support_size = seqs_len * neighbor_num[0]
        item_neighbors = [inputs, first_adj.view(batch_size, support_size)]

        for i in range(1, config.hop):
            item_sample_i = self.sample(item_neighbors[-1])
            support_size *= neighbor_num[i]
            item_neighbors.append(item_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]

        session_info = []
        for i in range(config.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))
        
        for n_hop in range(config.hop):
            entity_vectors_next_iter = []
            for hop in range(config.hop - n_hop):
                shape = [batch_size, -1, neighbor_num[hop], config.dim]
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, config.dim)


        # combine
        h_local = F.dropout(h_local, config.dropout_local, training=self.training)
        h_global = F.dropout(h_global, config.dropout_global, training=self.training)
        
        # alpha = torch.sigmoid(torch.matmul(h_local, self.w_s) + torch.matmul(h_global, self.w_g))
        # output =  alpha*h_local + (1-alpha)*h_global
        output = h_local + h_global
        return output

