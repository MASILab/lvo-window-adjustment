import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from data_loader import LvoDataLoader
from parallel_fc import ParallelFC
import torch.optim as optim
from utils import AverageMeter
import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import math


train_level = False
train_width = False
train_both = False
train_both_resplit = False
train_both_resplit_3fc = True
train_both_resplit_3fc_34 = False

if train_both_resplit_3fc:
    data_to_load = 'csv/resplit_dataset.csv'
    root_dir = 'results/window_both_resplit_3fc_mt_reg'
    save_epochs_dir = 'results/window_both_resplit_3fc_mt_reg/epochs'
    save_model_dir = 'results/window_both_resplit_3fc_mt_reg/models'
    save_csv_dir = 'results/window_both_resplit_3fc_mt_reg'

elif train_both_resplit_3fc_34:
    data_to_load = 'csv/resplit_dataset.csv'
    root_dir = 'results/window_both_resplit_3fc_34_3e5_mt_reg'
    save_epochs_dir = 'results/window_both_resplit_3fc_34_3e5_mt_reg/epochs'
    save_model_dir = 'results/window_both_resplit_3fc_34_3e5_mt_reg/models'
    save_csv_dir = 'results/window_both_resplit_3fc_34_3e5_mt_reg'

if not os.path.exists(root_dir):
    os.mkdir(root_dir)
if not os.path.exists(save_epochs_dir):
    os.mkdir(save_epochs_dir)
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)
if not os.path.exists(save_csv_dir):
    os.mkdir(save_csv_dir)


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = LvoDataLoader(csv_file=data_to_load, transform=transform, mode='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)

    validate_set = LvoDataLoader(csv_file=data_to_load, transform=transform, mode='val')
    validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=4, shuffle=False, num_workers=4)

    model = get_model(2, 40, num_layers=18).cuda()
    learning_rate = 0.00003

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = math.inf

    train_loss_history = []
    val_loss_history = []
    epoch_history = []
    for epoch in range(20):
        epoch += 1
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, 20, learning_rate))
        train_loss = train(train_loader, model, optimizer, criterion)
        val_loss = validate(validate_loader, model, criterion, epoch)

        epoch_history.append('epoch_' + str(epoch))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        model_name = 'epoch_' + str(epoch) + '.pth'
        model_dir = os.path.join(save_model_dir, model_name)
        torch.save(model.state_dict(), model_dir)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_name = 'best_model' + '.pth'
            best_model_dir = os.path.join(save_model_dir, best_model_name)
            torch.save(model.state_dict(), best_model_dir)

    df = pd.DataFrame()
    df['epochs'] = epoch_history
    df['train_loss'] = train_loss_history
    df['val_loss'] = val_loss_history
    output_file = os.path.join(save_csv_dir, 'epochs_summary.csv')
    df.to_csv(output_file)


def get_model(n_classes, image_channels, num_layers=18):
    if num_layers == 18:
        model = torchvision.models.resnet18()
    else:
        model = torchvision.models.resnet34()
    for p in model.parameters():
        p.requires_grad = True
    inf = model.fc.in_features
    hidf = 256

    model.fc = ParallelFC(inf, hidf, n_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def train(train_loader, model, optimizer, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    tbar = tqdm(train_loader, desc='\r')

    model.train()
    for batch_idx, (_, inputs, targets) in enumerate(tbar):
        data_time.update(time.time() - end)

        inputs = inputs.float()
        targets = torch.stack((targets[0], targets[1])).T.float()
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        outputs = torch.stack((outputs[0].T[0], outputs[1].T[0])).T.float()
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        tbar.set_description('\r Train Loss: %.3f' % losses.avg)

    return losses.avg


def validate(valloader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    tbar = tqdm(valloader, desc='\r')
    end = time.time()

    with torch.no_grad():
        name_history = []
        pred_level_history = []
        pred_width_history = []
        target_level_history = []
        target_width_history = []
        for batch_idx, (names, inputs, targets) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.float()
            targets = torch.stack((targets[0], targets[1])).T.float()
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            outputs = torch.stack((outputs[0].T[0], outputs[1].T[0])).T.float()
            # prob_out = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)

            # measure loss
            losses.update(loss.item(), inputs.size(0))

            pred = outputs.T.cpu().numpy()
            pred_level = pred[0]
            pred_width = pred[1]
            targets = targets.T.cpu().numpy()
            target_level = targets[0]
            target_width = targets[1]

            name_history = np.concatenate((name_history, names), axis=0)
            pred_level_history = np.concatenate((pred_level_history, pred_level), axis=0)
            pred_width_history = np.concatenate((pred_width_history, pred_width), axis=0)
            target_level_history = np.concatenate((target_level_history, target_level), axis=0)
            target_width_history = np.concatenate((target_width_history, target_width), axis=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            tbar.set_description('\r %s Loss: %.3f' % ('Validation', losses.avg))

        df = pd.DataFrame()
        df['subj'] = name_history
        df['prediction_level'] = pred_level_history
        df['prediction_width'] = pred_width_history
        df['target_level'] = target_level_history
        df['target_width'] = target_width_history
        df.to_csv(os.path.join(save_epochs_dir, 'epoch_'+str(epoch)+'.csv'))

    return losses.avg


if __name__ == '__main__':
    main()
