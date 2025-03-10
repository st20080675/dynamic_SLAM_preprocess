import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # the Sequential name has to be 'vgg feature'.
        # the params name will be like feature.0.weight ,
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            # nn.MaxPool2d((3, 3), (1, 1), (1, 1), ceil_mode=True),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            # nn.MaxPool2d((3, 3), (1, 1), (1, 1), ceil_mode=True),
            # AvgPool2d,
            # nn.AvgPool2d((3, 3), (1, 1), (1, 1), ceil_mode=True),
            nn.Conv2d(512, 1024, (3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, (3, 3), padding=1),
            # nn.ReLU(),
            nn.Dropout(0.5)
            # nn.AdaptiveAvgPool2d(1)
            # nn.Conv2d(1024,21,(1, 1)) # 1024 / 512
            # nn.Softmax2d()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_conv = nn.Conv2d(1024, 21, (1, 1))
        # self.softmax2d = nn.Softmax2d()
        # self.min_prob = 0.0001

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fc_feature = self.features(x)
        x = self.avg_pool(fc_feature)
        x = self.fc_conv(x)
        # sm_mask = self.softmax2d(fc_mask)+self.min_prob
        # sm_mask = sm_mask / sm_mask.sum(dim=1, keepdim=True)
        # sm_mask = self.softmax2d(sm_mask)

        return fc_feature, x

    def forward_cue(self, x):
        fc_feature = self.features(x)
        x = self.fc_conv(fc_feature)
        return fc_feature, x
