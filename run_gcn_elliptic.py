import argparse
from baseline import PYG_GCN
from utils.load_data import load_data, load_pyg_data
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


def train(data, model, criterion, optimizer):
    idx_train = data.train_mask
    idx_val = data.test_mask

    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)

    # acc_train = accuracy(output[idx_train], y[idx_train])
    loss_train = criterion(output[idx_train], data.y[idx_train])
    pred_train = output[idx_train].max(dim=1)[1]
    tp_train = ((pred_train == 1) & (data.y[idx_train] == 1)).sum()
    tn_train = ((pred_train == 0) & (data.y[idx_train] == 0)).sum()
    fp_train = ((pred_train == 1) & (data.y[idx_train] == 0)).sum()
    fn_train = ((pred_train == 0) & (data.y[idx_train] == 1)).sum()
    precision_val = tp_train.item() / (tp_train.item() + fp_train.item() + 10e-13)
    recall_val = tp_train.item() / (tp_train.item() + fn_train.item() + 10e-13)
    train_f1 = (2 * precision_val * recall_val) / (precision_val + recall_val + 10e-13)
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), train_f1


def validate(data, model, criterion):

    idx_train = data.train_mask
    idx_val = data.test_mask

    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        pred_val = output[idx_val].max(dim=1)[1]
        tp_val = ((pred_val == 1) & (data.y[idx_val] == 1)).sum()
        tn_val = ((pred_val == 0) & (data.y[idx_val] == 0)).sum()
        fp_val = ((pred_val == 1) & (data.y[idx_val] == 0)).sum()
        fn_val = ((pred_val == 0) & (data.y[idx_val] == 1)).sum()
        precision_val = tp_val.item() / (tp_val.item() + fp_val.item() + 10e-13)
        recall_val = tp_val.item() / (tp_val.item() + fn_val.item() + 10e-13)
        val_f1 = (2 * precision_val * recall_val) / (precision_val + recall_val + 10e-13)
        loss_val = criterion(output[idx_val], data.y[idx_val])
        return loss_val.item(), val_f1


def test(data, model, criterion, checkpt_file):
    idx_train = data.train_mask
    idx_val = data.val_mask
    idx_test = data.test_mask

    model.load_state_dict(torch.load(checkpt_file))
    model.eval()

    with torch.no_grad():
        output = model(data.x, data.edge_index)
        pred_test = output[idx_test].max(dim=1)[1]
        loss_test = criterion(output[idx_test], data.y[idx_test])
        tp_test = ((pred_test == 1) & (data.y[idx_test] == 1)).sum()
        tn_test = ((pred_test == 0) & (data.y[idx_test] == 0)).sum()
        fp_test = ((pred_test == 1) & (data.y[idx_test] == 0)).sum()
        fn_test = ((pred_test == 0) & (data.y[idx_test] == 1)).sum()
        precision_test = tp_test.item() / (tp_test.item() + fp_test.item() + 10e-13)
        recall_test = tp_test.item() / (tp_test.item() + fn_test.item() + 10e-13)
        val_test = (2 * precision_test * recall_test) / (precision_test + recall_test + 10e-13)

        return loss_test.item(), val_test

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='elliptic', help='Dataset')
    parser.add_argument('--epochs', type=int, default=1000, help='Max epoch.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')
    parser.add_argument('--hidden', type=int, default=100, help='Hidden dimension')
    parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
    parser.add_argument('--print-interval', type=int, default=20, help='Print Interval')
    parser.add_argument('--fix', action='store_true', default=True, help='fix split')
    args = parser.parse_args()

    data = load_pyg_data(args.dataset, fixed_split=args.fix)
    data = data.to(device)

    net = PYG_GCN(in_channels=data.x.shape[1], out_channels=args.hidden, num_class=2)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    checkpt_file = 'pt/model_{}_2.pt'.format(args.dataset)
    weights = torch.Tensor([1., 2])
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))


    c = 0
    best = 7e7
    best_epoch = 0
    acc = 0
    f1_best = 0

    for epoch in range(args.epochs):
        loss_tra, f1_train = train(data, net, criterion, optimizer)
        loss_val, f1_val = validate(data, net, criterion)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'loss:{:.3f}'.format(loss_tra),
                  'F1:{:.2f}'.format(f1_train * 100),
                  'loss:{:.3f}'.format(loss_val),
                  'F1:{:.2f}'.format(f1_val * 100),
                  'Best F1:{:.2f}'.format(f1_best * 100))
        if f1_val > f1_best:
            f1_best = f1_val

    return f1_best

if __name__ == '__main__':
    f1_list = []
    for _ in range(1):
        f1_list.append(main())
    print(np.mean(f1_list))