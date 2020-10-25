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
from utils import AverageMeter

train_both_resplit = False
train_both_resplit_3fc = False
train_both_resplit_3fc_34_3e5 = False

if train_both_resplit:
    data_to_load = 'csv/resplit_dataset.csv'
    best_model_dir = 'results/window_both_resplit_mt_reg/models/best_model.pth'
    test_result_dir = 'results/window_both_resplit_mt_reg'

elif train_both_resplit_3fc:
    data_to_load = 'csv/resplit_dataset.csv'
    best_model_dir = 'results/window_both_resplit_3fc_mt_reg/models/best_model.pth'
    test_result_dir = 'results/window_both_resplit_3fc_mt_reg'

elif train_both_resplit_3fc_34_3e5:
    data_to_load = 'csv/resplit_dataset.csv'
    best_model_dir = 'results/window_both_resplit_3fc_34_3e5_mt_reg/models/best_model.pth'
    test_result_dir = 'results/window_both_resplit_3fc_34_3e5_mt_reg'


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    test_set = LvoDataLoader(csv_file=data_to_load, transform=transform, mode='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    model = get_model(2, 40).cuda()
    model.load_state_dict(torch.load(best_model_dir))

    model.eval()
    criterion = nn.MSELoss()
    losses = AverageMeter()
    tbar = tqdm(test_loader, desc='\r')
    with torch.no_grad():
        name_history = []
        pred_level_history = []
        pred_width_history = []
        target_level_history = []
        target_width_history = []
        for batch_idx, (names, inputs, targets) in enumerate(tbar):

            inputs = inputs.float()
            targets = torch.stack((targets[0], targets[1])).T.float()
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)
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

            tbar.set_description('\r %s Loss: %.3f' % ('Test', losses.avg))

        df = pd.DataFrame()
        df['subj'] = name_history
        df['prediction_level'] = pred_level_history
        df['prediction_width'] = pred_width_history
        df['target_level'] = target_level_history
        df['target_width'] = target_width_history
        df.to_csv(os.path.join(test_result_dir, 'test.csv'))


def get_model(n_classes, image_channels):
    model = torchvision.models.resnet18()
    for p in model.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                             nn.Linear(in_features=256, out_features=256),
                             nn.Linear(in_features=256, out_features=n_classes))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


if __name__ == '__main__':
    main()
