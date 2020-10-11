import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from data_loader import LvoDataLoader
import numpy as np
import tqdm
from tqdm import tqdm
import pandas as pd
import os

train_level = True
train_width = not train_level

if train_level:
    data_to_load = 'csv/dataset_reg_level.csv'
    best_model_dir = 'results/window_level_reg/models/best_model.pth'
    test_result_dir = 'results/window_level_reg'

elif train_width:
    data_to_load = 'csv/dataset_reg_width.csv'
    best_model_dir = 'results/window_width_reg/models/best_model.pth'
    test_result_dir = 'results/window_width_reg'


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    test_set = LvoDataLoader(csv_file=data_to_load, transform=transform, mode='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    model = get_model(1, 40).cuda()
    model.load_state_dict(torch.load(best_model_dir))

    model.eval()
    tbar = tqdm(test_loader, desc='\r')
    with torch.no_grad():

        name_history = []
        pred_history = []
        target_history = []
        for batch_idx, (names, inputs, targets) in enumerate(tbar):

            inputs = inputs.float()
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets = targets.view(-1, 1).float()
            # compute output
            outputs = model(inputs)

            pred = torch.reshape(outputs, (-1, )).cpu().numpy()
            targets = torch.reshape(targets, (-1,)).cpu().numpy()

            name_history = np.concatenate((name_history, names), axis=0)
            pred_history = np.concatenate((pred_history, pred), axis=0)
            target_history = np.concatenate((target_history, targets), axis=0)

            # measure elapsed time
        df = pd.DataFrame()
        df['subj'] = name_history
        df['prediction'] = pred_history
        df['target'] = target_history
        df.to_csv(os.path.join(test_result_dir, 'test.csv'))


def get_model(n_classes, image_channels):
    model = torchvision.models.resnet18()
    for p in model.parameters():
        p.requires_grad = True
    inft = 512
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


if __name__ == '__main__':
    main()
