import torch
from torch import nn


class SemanticImageExtractor(nn.Module):
    """
    This class expected image as input with size (224x224x3)
    """

    def __init__(self, output_class_num, feature_size=200):
        super(SemanticImageExtractor, self).__init__()
        self.alx_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.alx_layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.alx_layer3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.alx_layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.alx_layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        # return the same number of features but change width and height of img

        self.fc06 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU()
        )

        self.fc07 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.ReLU()
        )

        self.fc08 = nn.Sequential(
            nn.Linear(feature_size, output_class_num),
            nn.Softmax())

    def forward(self, x):
        x = self.alx_layer1(x)
        x = self.alx_layer2(x)
        x = self.alx_layer3(x)
        x = self.alx_layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc06(x)
        semantic_features = self.fc07(x)
        p_label = self.fc08(semantic_features)
        return semantic_features, p_label


class SemanticEEGExtractor(nn.Module):
    def __init__(self, expected_shape: torch.Tensor, output_class_num: int, feature_size=200):
        """
        expected_shape [Batch_size, eeg_features, eeg_channel, sample_len]
        """
        super(SemanticEEGExtractor, self).__init__()

        self.batch_norm = nn.BatchNorm2d(num_features=1)

        self.fc01 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(expected_shape.shape[1] * expected_shape.shape[2], 4096),
            nn.LeakyReLU()
        )

        self.fc02 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size)
        )

        self.fc02_act = nn.Tanh()

        self.fc03 = nn.Sequential(
            nn.Linear(feature_size, output_class_num),
            nn.Softmax())

    def forward(self, eeg: torch.Tensor):
        eeg = eeg.unsqueeze(1)
        x = self.batch_norm(eeg)
        if torch.isnan(self.batch_norm.weight).item():
            print("<X> : NAN detected in batch_norm weight")
            exit()
        x = x.reshape([x.shape[0], -1])
        x = self.fc01(x)
        semantic_features = self.fc02(x)
        x = self.fc02_act(semantic_features)
        label = self.fc03(x)
        return semantic_features, label


class Generator(nn.Module):  # <<- CGAN
    # How can we input both label and features?
    EXPECTED_NOISE = 2064  # << For EEGImageNet with 48x48

    # EXPECTED_NOISE = 2101  # Cylinder_RGB with 48x48

    def __init__(self, num_classes):
        super(Generator, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1, 256, kernel_size=5, stride=1, bias=False),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, bias=False),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, bias=False),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, bias=False),
            nn.ReLU()
        )
        # self.deconv5 = nn.Sequential(
        #     nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, output_padding=1),
        #     nn.ReLU()
        # )
        self.num_classes = num_classes

    def __forward_check(self, z, eeg_semantic, eeg_label):
        if z.shape[1] != Generator.EXPECTED_NOISE:
            raise RuntimeError("Incorrect shape of vector \'z\'")
        if eeg_semantic.shape[1] != 200:
            raise RuntimeError("Incorrect shape of vector \'eeg_semantic\'")
        if eeg_label.shape[1] != self.num_classes:
            raise RuntimeError("Incorrect shape of vector \'eeg_label\'")

    # -- Expected shape --
    # z.shape = (3839,)
    # eeg_semantic.shape = (200,)
    # label.shape = (10,)
    def forward(self, z, semantic, label):
        # First, we need to concat.
        # Problem
        #   Should we concat and deconvolution it?
        #   Second problem, what is the size of z
        self.__forward_check(z, semantic, label)
        x = torch.cat((z, semantic, label), 1)
        x = x.unsqueeze(1).unsqueeze(1)
        x = x.reshape(x.shape[0], 1, 48, 48)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        # x = self.deconv5(x)
        return x


# Should D1 and D2 takes an real/gen image as an input?
# D1 : Image only
# D2 : Semantic features and label
class D1(nn.Module):
    def __init__(self):
        super(D1, self).__init__()
        self.conv1 = nn.Sequential(  # Currently we input black and white img
            nn.BatchNorm2d(num_features=6),
            nn.Conv2d(6, 64, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=46),
            nn.LeakyReLU(),
            nn.Linear(in_features=46, out_features=1),
            # nn.Sigmoid()
        )

    @staticmethod
    def __forward_check(x):
        shape = (x.shape[1], x.shape[2], x.shape[3])
        if shape != (3, 64, 64):
            raise RuntimeError("Expected shape", (3, 64, 64))

    def forward(self, x1, x2):
        '''

        :param x1: First image to input
        :param x2: Second image to input but... How we gonna concat? concat in channel dim? YES I THINK WE CAN!
        :return: real or not
        '''
        self.__forward_check(x1)
        self.__forward_check(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        x = self.final_fc(x)
        return x


# In the paper, This is D1
class D2(nn.Module):
    def __init__(self):
        super(D2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )

        self.final_fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(12784, 226),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(226, 1),
            nn.Sigmoid()
        )

        self.final_fc_cylinder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(12747, 226),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(226, 1),
            nn.Sigmoid()
        )

        self.final_fc2 = nn.Sequential(
            nn.Linear(512, 46),
            nn.LeakyReLU(),
            nn.Linear(46, 1),
            # nn.Sigmoid()
        )

    @staticmethod
    def __forward_check(img):  # , eeg_features, eeg_label):
        # if eeg_features.shape[1] != 200:
        #     raise RuntimeError("Expected features size = 200")
        # if eeg_label.shape[1] != 569:
        #     raise RuntimeError("Expected shape size = 569")
        img_shape = (img.shape[1], img.shape[2], img.shape[3])
        if img_shape != (3, 64, 64):
            raise RuntimeError("Expected shape", (3, 64, 64))

    def forward(self, img, features, label):  # , eeg_features, eeg_label):
        # self.__forward_check(img, eeg_features, eeg_label)
        x = self.conv1(img)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)

        x = x.flatten(start_dim=1)
        x = torch.cat((x, features), 1)  # Concat eeg_features
        x = torch.cat((x, label), 1)  # Concat label
        x = self.final_fc(x)
        # x = self.final_fc_cylinder(x)
        return x


def __test_execution():
    sample = torch.rand(1, 3, 64, 64)
    model = D2()
    out = model(sample)
