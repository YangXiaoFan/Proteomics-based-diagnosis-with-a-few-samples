# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif



def train(config, model, set_1_train_iter, set_1_dev_iter, set_2_train_iter, set_2_test_iter, loss_weight = [1.2, 0.8]):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.weight_decay)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, last_epoch=-1)
    dev_best_loss = float('inf')
    writer = open(config.log_path + '_' + time.strftime('%m-%d_%H.%M', time.localtime()) + '.txt', "w")
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}], learning rate = {}'.format(epoch + 1, config.num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
        writer.write('Epoch [{}/{}], learning rate = {}\n'.format(epoch + 1, config.num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
        for i, (trains, labels) in enumerate(set_1_train_iter):
            outputs_1, outputs_2 = model(trains)
            model.zero_grad()
            loss_1 = F.cross_entropy(outputs_1, labels[0])
            loss_2 = F.smooth_l1_loss(torch.mul(outputs_2, labels[2]), labels[1])
            loss = loss_weight[0] * loss_1 + loss_weight[1] * loss_2
            loss.backward()
            optimizer.step()

        true = labels[0].data.cpu()
        predic = torch.max(outputs_1.data, 1)[1].cpu()
        train_acc = metrics.accuracy_score(true, predic)
        dev_acc, dev_loss, dev_loss_1, dev_loss_2 = evaluate(model, set_1_dev_iter)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        msg = 'epoch: {0:>6},  Train Loss: {1:>5.3},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.3},  Val Acc: {4:>6.2%},  Loss 1: {5:>5.3},  Loss 2: {6:>5.3},  Time: {7} {8}'
        print(msg.format(epoch, loss.item(), train_acc, dev_loss, dev_acc, dev_loss_1, dev_loss_2, time_dif, improve))
        msg = 'epoch: {0:>6},  Train Loss: {1:>5.3},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.3},  Val Acc: {4:>6.2%},  Time: {5} {6}\n'
        writer.write(msg.format(epoch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

        model.train()

        scheduler.step() # 学习率衰减

    test_acc, _, _ = test(config, model, set_2_train_iter, set_2_test_iter)
    msg = 'Test Acc: {0:>6.2%}'
    print(msg.format(test_acc))
    msg = 'Test Acc: {0:>6.2%}\n'
    writer.write(msg.format(test_acc))    

    writer.close()


def test(config, model, set_2_train_iter, set_2_test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    t_prototype = []
    n_prototype = []

    with torch.no_grad():
        for texts, labels in set_2_train_iter:
            model(texts)
            labels = labels.data.cpu().numpy()
            for i in range(len(labels)):
                if labels[i] == 1:
                    embed = model.embed.cpu().numpy()[i,:,:]
                    t_prototype.append(embed.reshape(-1))
                else:
                    embed = model.embed.cpu().numpy()[i,:,:]
                    n_prototype.append(embed.reshape(-1))

    t_prototype = np.average(t_prototype, axis=0)
    n_prototype = np.average(n_prototype, axis=0)
    predict_all = []
    labels_all = []

    with torch.no_grad():
        for texts, labels in set_2_test_iter:
            model(texts)
            labels = labels.data.cpu().numpy()
            for i in range(len(labels)):
                embed = model.embed.cpu().numpy()[i,:,:].reshape(-1)
                t_dist = np.sum(np.dot(t_prototype, embed)) / (np.linalg.norm(t_prototype) * np.linalg.norm(embed))
                n_dist = np.sum(np.dot(n_prototype, embed)) / (np.linalg.norm(n_prototype) * np.linalg.norm(embed))

                if t_dist > n_dist:
                    predict_all.append(1)
                else:
                    predict_all.append(0)
                labels_all.append(labels[i])

    acc = metrics.accuracy_score(labels_all, predict_all)

    return acc, labels_all, predict_all
    

def evaluate(model, data_iter, loss_weight = [1.2, 0.8]):
    model.eval()
    loss_total = 0
    loss_1_total = 0
    loss_2_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs_1, outputs_2 = model(texts)
            loss_1 = F.cross_entropy(outputs_1, labels[0])
            loss_2 = F.smooth_l1_loss(torch.mul(outputs_2, labels[2]), labels[1])
            loss = loss_weight[0] * loss_1 + loss_weight[1] * loss_2
            loss_total += loss
            loss_1_total += loss_weight[0] * loss_1
            loss_2_total += loss_weight[1] * loss_2
            labels = labels[0].data.cpu().numpy()
            predic = torch.max(outputs_1.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data_iter), loss_1_total / len(data_iter), loss_2_total / len(data_iter)