import torch.nn as nn


class ParallelFC(nn.Module):
    def __init__(self, in_features, hid_features, num_classes):
        super(ParallelFC, self).__init__()
        self.level_fc = nn.Sequential(nn.Linear(in_features=in_features, out_features=num_classes))

        self.width_fc = nn.Sequential(nn.Linear(in_features=in_features, out_features=hid_features),
                                      nn.Linear(in_features=in_features, out_features=num_classes))

    def forward(self, x):
        level = self.level_fc(x)
        width = self.width_fc(x)

        return level, width


