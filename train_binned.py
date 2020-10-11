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
import torch.nn.functional as F
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import os

transform = transforms.Compose([transforms.ToTensor()])

train_set = LvoDataLoader(csv_file='csv/dataset_bin.csv', transform=transform, mode='train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)

validate_set = LvoDataLoader(csv_file='csv/dataset_bin.csv', transform=transform, mode='val')
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=4, shuffle=False, num_workers=4)

test_set = LvoDataLoader(csv_file='csv/dataset_bin.csv', transform=transform, mode='test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)


def get_model(n_classes, image_channels):
    model = torchvision.models.resnet18()
    for p in model.parameters():
        p.requires_grad = True
    inft = model.fc.in_features
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


model = get_model(2, 40).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train(train_loader, model, optimizer, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    top1 = AverageMeter()
    tbar = tqdm(train_loader, desc='\r')

    model.train()
    for batch_idx, (inputs, targets) in enumerate(tbar):
        data_time.update(time.time() - end)

        inputs = inputs.float()
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        [acc1, ] = accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        tbar.set_description('\r Train Loss: %.3f | Top1: %.3f' % (losses.avg, top1.avg))

    return losses.avg, top1.avg


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

        pred_history = []
        target_history = []
        for batch_idx, (inputs, targets) in enumerate(tbar):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.float()
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            # prob_out = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            [prec1,] = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            pred = F.softmax(outputs, dim=1)
            pred = pred.data.max(1)[1]

            pred_history = np.concatenate((pred_history, pred.data.cpu().numpy()), axis=0)
            target_history = np.concatenate((target_history, targets.data.cpu().numpy()), axis=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            tbar.set_description('\r %s Loss: %.3f | Top1: %.3f' % ('Validation', losses.avg, top1.avg))

        df = pd.DataFrame()
        df['prediction'] = pred_history
        df['target'] = target_history
        df.to_csv(os.path.join('epochs', 'epoch_'+str(epoch)+'.csv'))

    return losses.avg, top1.avg


for epoch in range(20):
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, 10, 0.0001))
    train(train_loader, model, optimizer, criterion)
    validate(validate_loader, model, criterion, epoch)

