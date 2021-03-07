import argparse
from module import NodeBernNet
from utils.load_data import load_data
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from scipy import special

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_dir = os.path.expanduser('~') + '/log/'

def train(data, model, optimizer, criterion):
    x, y, laplacian, mask = data
    idx_train, idx_test = mask

    anomaly = (y == 1)
    normal = (y == 0)
    index = torch.full(y.size(), False, device=device)
    index[idx_train] = True
    anomaly_train = index & anomaly
    normal_train = index & normal

    model.train()
    optimizer.zero_grad()
    output, bias_loss = model(x, laplacian, label=(anomaly_train, normal_train))
    # acc_train = accuracy(output[idx_train], y[idx_train])
    loss_train = criterion(output[idx_train], y[idx_train])
    loss_train += bias_loss * .1

    loss_train.backward()
    optimizer.step()


    pred_train = output[idx_train].max(dim=1)[1]
    tp_train = ((pred_train == 1) & (y[idx_train] == 1)).sum()
    tn_train = ((pred_train == 0) & (y[idx_train] == 0)).sum()
    fp_train = ((pred_train == 1) & (y[idx_train] == 0)).sum()
    fn_train = ((pred_train == 0) & (y[idx_train] == 1)).sum()
    precision_val = tp_train.item() / (tp_train.item() + fp_train.item() + 10e-13)
    recall_val = tp_train.item() / (tp_train.item() + fn_train.item() + 10e-13)
    train_f1 = (2 * precision_val * recall_val) / (precision_val + recall_val + 10e-13)

    return loss_train.item(), train_f1

def validate(data, model, criterion):
    x, y, laplacian, mask = data
    idx_train, idx_val = mask

    model.eval()
    with torch.no_grad():
        output = model(x, laplacian)
        pred_val = output[idx_val].max(dim=1)[1]
        tp_val = ((pred_val == 1) & (y[idx_val] == 1)).sum()
        tn_val = ((pred_val == 0) & (y[idx_val] == 0)).sum()
        fp_val = ((pred_val == 1) & (y[idx_val] == 0)).sum()
        fn_val = ((pred_val == 0) & (y[idx_val] == 1)).sum()
        precision_val = tp_val.item() / (tp_val.item() + fp_val.item() + 10e-13)
        recall_val = tp_val.item() / (tp_val.item() + fn_val.item() + 10e-13)
        val_f1 = (2 * precision_val * recall_val) / (precision_val + recall_val + 10e-13)
        loss_val = criterion(output[idx_val], y[idx_val])
        return loss_val.item(), val_f1


def test(args, data, model, criterion, checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    x, y, laplacian, mask = data
    idx_train, idx_test = mask

    model.eval()

    with torch.no_grad():
        output, attn_score = model(x, laplacian, exp=True)

        anomaly = (y == 1).cpu()
        normal = (y == 0).cpu()
        index = torch.full(y.size(), False)
        index[idx_test] = True
        logger.info("Test:")
        ab_f0  = torch.mean(attn_score[index & anomaly, 0]).item()
        ab_f1  = torch.mean(attn_score[index & anomaly, 1]).item()
        n_f0  = torch.mean(attn_score[index & normal, 0]).item()
        n_f1  = torch.mean(attn_score[index & normal, 1]).item()

        logger.info("Anomaly Filter0 {}".format(ab_f0))
        logger.info("Anomaly Filter1 {}".format(ab_f1))
        logger.info("Normal Filter0 {}".format(n_f0))
        logger.info("Normal Filter1  {}".format(n_f1))

        name_list = ['Anomaly Filter0', 'Filter1', 'Normal Filter0', 'Filter1']
        num_list = [ab_f0, ab_f1, n_f0, n_f1]
        plt.bar(range(len(num_list)), num_list, tick_label=name_list)
        plt.show()


        index = torch.full(y.size(), False)
        index[idx_train] = True
        logger.info("Train:")
        logger.info("Anomaly Filter1 {}".format(torch.mean(attn_score[index & anomaly, 0])))
        logger.info("Anomaly Filter2 {}".format(torch.mean(attn_score[index & anomaly, 1])))
        logger.info("Normal Filter1 {}".format(torch.mean(attn_score[index & normal, 0])))
        logger.info("Normal Filter2 {}".format(torch.mean(attn_score[index & normal, 1])))


        pred_test = output[idx_test].max(dim=1)[1]
        loss_test = criterion(output[idx_test], y[idx_test])
        tp_test = ((pred_test == 1) & (y[idx_test] == 1)).sum()
        tn_test = ((pred_test == 0) & (y[idx_test] == 0)).sum()
        fp_test = ((pred_test == 1) & (y[idx_test] == 0)).sum()
        fn_test = ((pred_test == 0) & (y[idx_test] == 1)).sum()
        precision_test = tp_test.item() / (tp_test.item() + fp_test.item() + 10e-13)
        recall_test = tp_test.item() / (tp_test.item() + fn_test.item() + 10e-13)
        f1_test = (2 * precision_test * recall_test) / (precision_test + recall_test + 10e-13)




        logger.info('tp_test:{}\ttn_test:{}\tfp_test:{}\tfn_test:{}'.format(tp_test, tn_test, fp_test, fn_test))
        logger.info('precision_test:{}\trecall_test:{}\tf1_test{}'.format(precision_test, recall_test, f1_test))


        if args.K == 3:
            for filter_ in model.filters:
                theta_list = [filter_.theta0.item(), filter_.theta1.item(),
                              filter_.theta2.item(), filter_.theta3.item()]
                if args.softmax:
                    theta_list = special.softmax(theta_list)
                theta0, theta1, theta2, theta3 = theta_list
                logger.info('theta0:{}\ttheta1:{}\ttheta2:{}\ttheta3:{}\t'.format(theta0, theta1, theta2, theta3))
                coef = [-theta0 + 3. * theta1 - 3. * theta2 + theta3,
                        3. * theta0 - 6. * theta1 + 3. * theta2,
                        -3. * theta0 + 3. * theta1,
                        theta0]
                bernstein = np.poly1d(coef)
                x_line = np.linspace(0, 1, 1000)
                plt.plot(x_line, abs(bernstein(x_line)))
                plt.title(args.dataset + ' F1:{:.2f} '.format(f1_test) + args.normalization)
                plt.show()

        elif args.K == 4:
            for index, filter_ in enumerate(model.filters):
                theta_list = [filter_.theta0.item(), filter_.theta1.item(),
                              filter_.theta2.item(), filter_.theta3.item(),
                              filter_.theta4.item()]
                if args.softmax:
                    theta_list = special.softmax(theta_list)
                theta0, theta1, theta2, theta3, theta4 = theta_list
                logger.info('theta0:{}\ttheta1:{}\ttheta2:{}\ttheta3:{}\ttheta4:{}\t'.format(theta0, theta1, theta2, theta3, theta4))
                coef = [theta0 - 4. * theta1 + 6. * theta2 - 4. * theta3 + theta4,
                        -4. * theta0 + 12. * theta1 - 12. * theta2 + 4. * theta3,
                        6. * theta0 - 12. * theta1 + 6. * theta2,
                        -4. * theta0 + 4. * theta1,
                        theta0]
                bernstein = np.poly1d(coef)
                x_line = np.linspace(0, 1, 1000)
                plt.plot(x_line, abs(bernstein(x_line)))
                plt.title('Filter {}'.format(index) + args.dataset + ' F1:{:.2f} '.format(f1_test) + args.normalization)
                plt.show()

        return loss_test.item(), f1_test


def main(args, exp_num):
    laplacian, x, y, mask = load_data(args.dataset, normalization=args.normalization,
                                      fixed_split=args.fix, split='0.7_0.3')

    x = x.to(device)
    y = y.to(device)
    laplacian = laplacian.to(device)
    data = (x, y, laplacian, mask)

    net = NodeBernNet(in_channels=x.shape[1], K=args.K, out_channels=args.hidden, num_class=2, attn=args.attn, softmax=args.softmax)
    net.to(device)

    opt_config = None
    if args.K == 3:
        opt_config = [
            {'params': net.filters[0].theta0, 'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            {'params': net.filters[1].theta3, 'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            #
            #
            {'params': net.filters[0].theta1, 'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            {'params': net.filters[1].theta2, 'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            #
            {'params': net.filters[0].theta2, 'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            {'params': net.filters[1].theta1, 'weight_decay': 1e-4, 'lr': args.lr / 1.0},

            {'params': net.filters[0].theta3, 'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            {'params': net.filters[1].theta0, 'weight_decay': 1e-4, 'lr': args.lr / 1.0},

            {'params': net.params_linear, 'weight_decay': args.wd, 'lr': args.lr / 12.0},
            {'params': net.params_attn, 'weight_decay': args.wd, 'lr': args.lr / 12.0}
        ]
    elif args.K == 4:
        opt_config = [
            {'params': net.filters[0].theta0,'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            {'params': net.filters[1].theta4,'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            #
            #
            {'params': net.filters[0].theta1, 'weight_decay': 1e-4,'lr': args.lr / 1.0},
            {'params': net.filters[1].theta3, 'weight_decay': 1e-4,'lr': args.lr / 1.0},
            #
            {'params': net.filters[0].theta2,'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            {'params': net.filters[1].theta2,'weight_decay': 1e-4, 'lr': args.lr / 1.0},

            {'params': net.filters[0].theta3, 'weight_decay': 1e-4,'lr': args.lr / 1.0},
            {'params': net.filters[1].theta1, 'weight_decay': 1e-4,'lr': args.lr / 1.0},

            {'params': net.filters[0].theta4,'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            {'params': net.filters[1].theta0,'weight_decay': 1e-4, 'lr': args.lr / 1.0},
            #
            {'params': net.params_linear, 'weight_decay': args.wd, 'lr': args.lr / 40.0},
            {'params': net.params_attn, 'weight_decay': args.wd, 'lr': args.lr / 40.0}
        ]

    optimizer = optim.Adam(opt_config)
    checkpt_file = 'pt/model_{}_{}_{}_{}_Bob.pt'.format(args.dataset, args.normalization, args.K, exp_num)

    # weights = torch.Tensor([10., ((y == 0).sum() / (y == 1).sum()).item()])
    weights = torch.Tensor([.3, .7])
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))


    c = 0
    best = 7e7
    best_epoch = 0
    acc = 0
    f1_best = -1

    for epoch in range(args.epochs):
        loss_tra, f1_train = train(data, net, optimizer, criterion)
        loss_val, f1_val = validate(data, net, criterion)
        if (epoch + 1) % args.print_interval == 0 or epoch == 0:
            logger.info(
                'Epoch:{:04d}\tloss:{:.3f}\tF1:{:.2f}\tloss:{:.3f}\tF1_val: {}\tF1 Best{:.2f}:'.
                    format(epoch + 1, loss_tra, f1_train * 100, loss_val, f1_val * 100, f1_best * 100))
        if f1_val > f1_best:
            f1_best = f1_val
            torch.save(net.state_dict(), checkpt_file)


    test(args, data, net, criterion, checkpt_file)

    return f1_best


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='elliptic', help='Dataset')
    parser.add_argument('--epochs', type=int, default=2000, help='Max epoch.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
    parser.add_argument('--print-interval', type=int, default=20, help='Print Interval')
    parser.add_argument('--K', type=int, default=4, help='K')
    parser.add_argument('--fix', action='store_true', default=True, help='fix split')
    parser.add_argument('--softmax', action='store_true', default=True)
    parser.add_argument('--normalization', type=str, default='Bern', help='Model')
    parser.add_argument('--logName', type=str, default='log', help='log')
    parser.add_argument('--attn', type=str, default='Bili', help='Model')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  #


    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


    fh = logging.FileHandler(log_dir+args.logName+'APANet'+'_'+str(args.K)+'_'+args.attn+'.log')
    fh.setLevel(logging.INFO)

    logger.addHandler(fh)



    f1_list = []
    for i in range(5):
        logger.info("Run {} times:".format(i))
        f1_list.append(main(args, exp_num=i))
    logger.info(f1_list)
    logger.info(np.mean(f1_list))