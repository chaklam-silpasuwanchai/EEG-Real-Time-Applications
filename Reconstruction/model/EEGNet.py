import torch
import torch.nn as nn
from layers import *

"""This model is reimplemented on pytorch followed by this paper : EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces"""


class EEGNet(nn.Module):
    def __init__(self, in_channel, samples, kern_len, F1, F2, D, nb_classes, dropout_type='2D', dropout_rate=0.5,
                 norm_rate=0.25):
        """
        Expected input shape (BS, 1 (Later will be temporal features), CH, Samples)
        :param in_channel:
        :param samples:
        :param kern_len:
        :param F1:
        :param F2:
        :param D:
        """
        super(EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(kernel_size=(1, kern_len), in_channels=1, out_channels=F1,
                      padding='same', bias=False),  # Shape=(BS,TEM=F1,CH,LEN=samples)
            nn.BatchNorm2d(num_features=F1),
            ConstraintConv2d(kernel_size=(in_channel, 1), in_channels=F1, groups=F1,
                             out_channels=D * F1, bias=False, weight_max_lim=1.0),
            # Depthwise : shape=(BS, TEM=D*F1, 1,LEN)
            nn.BatchNorm2d(num_features=D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4))
        )

        if dropout_type == '2D':
            self.dropout = nn.Dropout2d(dropout_rate)
        elif dropout_type == 'classics':
            self.dropout = nn.Dropout(dropout_rate)

        # Depthwise : shape=(BS, TEM=D*F1, 1,LEN//4)
        self.block2 = nn.Sequential(
            SeparableConv2d(kernel_size=(1, 16), in_channels=D * F1, out_channels=F2, padding='same', bias=False),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8))
        )

        self.final_fc = nn.Sequential(
            ConstraintLinear(in_features=F2 * (samples // 32), out_features=nb_classes, weight_max_lim=norm_rate),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.final_fc(x)
        return x


if __name__ == '__main__':
    BS, TEM, CH, LEN = 2, 1, 16, 128
    inp = torch.rand(BS, TEM, CH, LEN)
    model = EEGNet(in_channel=CH, samples=LEN, kern_len=LEN//2, F1=16, F2=32, D=2, nb_classes=2)
    out = model(inp)
    print(out)
