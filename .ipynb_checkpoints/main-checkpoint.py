import config
import time
import datetime
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from model import SessionGraph
from utils import handle_adj, Data

def init_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, epoch):
    alias_inputs, adj, items, mask, targets, inputs, first_adj = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    first_adj = trans_to_cuda(first_adj).long()

    hidden,hiddem_emb = model(items, adj, mask, inputs, first_adj)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    if epoch <=-1:
        return targets, model.compute_scores(seq_hidden, mask)
    else:
        return targets, model.compute_scores(seq_hidden +hiddem_emb, mask)


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=config.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data, epoch)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=config.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit20, mrr20 = [], []
    for data in test_loader:
        targets, scores = forward(model, data, epoch)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))

    hit10, mrr10 = [], []
    for data in test_loader:
        targets, scores = forward(model, data, epoch)
        sub_scores = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit10) * 100)
    result.append(np.mean(hit20) * 100)
    result.append(np.mean(mrr10) * 100)
    result.append(np.mean(mrr20) * 100)       

    return result

if __name__ == "__main__":
    print(config.dataset, config.num_node, "lr_dc:",config.lr_dc, "lr_dc_step:",config.lr_dc_step, "dropout_gcn:", config.dropout_gcn, "dropout_local:",config.dropout_local, "max_relative_position:",config.max_relative_position)
    init_seed(2021)

    train_data = pickle.load(open('datasets/' + config.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + config.dataset + '/test.txt', 'rb'))

    adj = pickle.load(open('datasets/' + config.dataset + '/adj_' + str(12) + '.pkl', 'rb'))
    seq2fre = pickle.load(open('datasets/' + config.dataset + '/seq2fre_' + str(12) + '.pkl', 'rb'))
    seq2idx = pickle.load(open('datasets/' + config.dataset + '/seq2idx_' + str(12) + '.pkl', 'rb'))

    train_data = Data(adj, seq2fre, seq2idx, train_data)
    test_data = Data(adj, seq2fre, seq2idx, test_data)

    adj = handle_adj(adj, 12)
    print(len(adj))
    model = trans_to_cuda(SessionGraph(trans_to_cuda(torch.Tensor(adj)).long()))

    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(config.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit10, hit20, mrr10, mrr20 = train_test(model, train_data, test_data, epoch)
        flag = 0
        if hit10 >= best_result[0]:
            best_result[0] = hit10
            best_epoch[0] = epoch
            flag = 1
        if hit20 >= best_result[1]:
            best_result[1] = hit20
            best_epoch[1] = epoch
            flag = 1
        if mrr10 >= best_result[2]:
            best_result[2] = mrr10
            best_epoch[2] = epoch
            flag = 1
        if mrr20 >= best_result[3]:
            best_result[3] = mrr20
            best_epoch[3] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f' % (hit10, hit20, mrr10, mrr20))
        print('Best Result:')
        print('\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d,\t%d\t%d' % (
            best_result[0], best_result[1], best_result[2], best_result[3], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3]))
        bad_counter += 1 - flag
        if bad_counter >= config.patience:
            break

        
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))