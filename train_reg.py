import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from data_loader import LvoDataLoader
import torch.optim as optim
from utils import AverageMeter, accuracy
import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import math

train_level = False
train_width = False
train_both = True
if train_level:
    data_to_load = 'csv/dataset_reg_level.csv'
    save_epochs_dir = 'results/window_level_reg/epochs'
    save_model_dir = 'results/window_level_reg/models'

elif train_width:
    data_to_load = 'csv/dataset_reg_width.csv'
    save_epochs_dir = 'results/window_width_reg/epochs'
    save_model_dir = 'results/window_width_reg/models'

elif train_both:
    data_to_load = 'csv/dataset.csv'
    save_epochs_dir = 'results/window_both_reg/epochs'
    save_model_dir = 'results/window_both_reg/models'


if train_level and train_width:
    assert not (train_width and train_level), 'Should not train both'


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = LvoDataLoader(csv_file=data_to_load, transform=transform, mode='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)

    validate_set = LvoDataLoader(csv_file=data_to_load, transform=transform, mode='val')
    validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=4, shuffle=False, num_workers=4)

    model = get_model(2, 40).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    best_loss = math.inf

    for epoch in range(10):
        epoch += 1
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, 10, 0.01))
        train(train_loader, model, optimizer, criterion)
        loss = validate(validate_loader, model, criterion, epoch)

        model_name = 'epoch_' + str(epoch) + '.pth'
        model_dir = os.path.join(save_model_dir, model_name)
        torch.save(model.state_dict(), model_dir)

        if loss < best_loss:
            best_loss = loss
            best_model_name = 'best_model' + '.pth'
            best_model_dir = os.path.join(save_model_dir, best_model_name)
            torch.save(model.state_dict(), best_model_dir)


def get_model(n_classes, image_channels):
    model = torchvision.models.resnet18()
    for p in model.parameters():
        p.requires_grad = True
    inft = 512
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def train(train_loader, model, optimizer, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    top1 = AverageMeter()
    tbar = tqdm(train_loader, desc='\r')

    model.train()
    for batch_idx, (_, inputs, targets) in enumerate(tbar):
        data_time.update(time.time() - end)

        inputs = inputs.float()
        targets = torch.stack((targets[0], targets[1])).T.float()
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        outputs = outputs.float()
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        [acc1, ] = accuracy(outputs, targets, topk=(1,))
        top1.update(acc1.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        tbar.set_description('\r Train Loss: %.3f | Top1: %.3f' % (losses.avg, top1.avg))

    return losses.avg


def validate(valloader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    tbar = tqdm(valloader, desc='\r')
    end = time.time()

    with torch.no_grad():
        name_history = []
        pred_history = []
        target_history = []
        for batch_idx, (names, inputs, targets) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.float()
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets = targets.view(-1, 1).float()
            # compute output
            outputs = model(inputs)
            # prob_out = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            [prec1,] = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            pred = torch.reshape(outputs, (-1, )).cpu().numpy()
            targets = torch.reshape(targets, (-1,)).cpu().numpy()

            name_history = np.concatenate((name_history, names), axis=0)
            pred_history = np.concatenate((pred_history, pred), axis=0)
            target_history = np.concatenate((target_history, targets), axis=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            tbar.set_description('\r %s Loss: %.3f | Top1: %.3f' % ('Validation', losses.avg, top1.avg))

        df = pd.DataFrame()
        df['subj'] = name_history
        df['prediction'] = pred_history
        df['target'] = target_history
        df.to_csv(os.path.join(save_epochs_dir, 'epoch_'+str(epoch)+'.csv'))

    return losses.avg


if __name__ == '__main__':
    main()
