import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings
from torch.nn.utils import spectral_norm

""" Yang, Z. et al. Comprehensive competition mechanism in palmprint recognition. IEEE Trans. Inf. Forensics Secur. 18, 5160–5170 (2023).
    https://github.com/Zi-YuanYang/CCNet
"""

class GaborConv2d(nn.Module):
    '''
    DESCRIPTION: an implementation of the Learnable Gabor Convolution (LGC) layer \n
    INPUTS: \n
    channel_in: should be 1 \n
    channel_out: number of the output channels \n
    kernel_size, stride, padding: 2D convolution parameters \n
    init_ratio: scale factor of the initial parameters (receptive filed) \n
    '''

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()

        # assert channel_in == 1

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.init_ratio = init_ratio

        self.kernel = 0

        if init_ratio <= 0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # initial parameters
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        # shape & scale of the Gaussian functioin:
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
                                  requires_grad=False)

        # frequency of the cosine envolope:
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()

        # [channel_out, channelin, kernel_H, kernel_W]
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # x=x.float()
        # y=y.float()

        # Rotated coordinate systems
        # [channel_out, <channel_in, kernel, kernel>], broadcasting
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8 * sigma.view(-1, 1, 1, 1) ** 2)) \
             * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

        gb = gb - gb.mean(dim=[2, 3], keepdim=True)

        return gb

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma,
                                   self.theta, self.f, self.psi)
        self.kernel = kernel
        # print(x.shape)
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    '''
    DESCRIPTION: an implementation of the Competitive Block::

    [CB = LGC + argmax + PPU] \n

    INPUTS: \n

    channel_in: only support 1 \n
    n_competitor: number of channels of the LGC (channel_out)  \n

    ksize, stride, padding: 2D convolution parameters \n

    init_ratio: scale factor of the initial parameters (receptive filed) \n

    o1, o2: numbers of channels of the conv_1 and conv_2 layers in the PPU, respectively. (PPU parameters)
    '''

    def __init__(self, channel_in, n_competitor, ksize, stride, padding, weight, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock_Mul_Ord_Comp, self).__init__()

        # assert channel_in == 1
        self.channel_in = channel_in
        self.n_competitor = n_competitor

        self.init_ratio = init_ratio

        self.gabor_conv2d = GaborConv2d(channel_in=channel_in, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                        padding=ksize // 2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(channel_in=n_competitor, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                         padding=ksize // 2, init_ratio=init_ratio)
        ## 2 2 no conv layer
        # soft-argmax
        # self.a = nn.Parameter(torch.FloatTensor([1]))
        # self.b = nn.Parameter(torch.FloatTensor([0]))

        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        # PPU
        self.conv1_1 = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.weight_chan = weight
        self.weight_spa = (1 - weight) / 2
        # print(self.weight_chan)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        # 1-st order
        x = self.gabor_conv2d(x)
        # print(x.shape)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        # print(x1_1.dtype)
        # print(x1_2.dtype)
        # print(x1_3.dtype)
        # print(x1_1.device)
        # print(x1_2.device)
        # print(x1_3.device)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)

        x_1 = self.se1(x_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        # 2-nd order
        x = self.gabor_conv2d2(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)
        x_2 = self.se2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1.view(x_1.shape[0], -1), x_2.view(x_2.shape[0], -1)), dim=1)

        return xx


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance::
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)

        From: https://github.com/ronghuaiyang/arcface-pytorch
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            '''
            print("Size of one_hot:", one_hot.size())
            print("Size of label:", label.size())
            print("Type of one_hot:", type(one_hot))
            print("Type of label:", type(label))
            unique_values = torch.unique(label)
            max_value = torch.max(unique_values)
            print("All values in label:", label)
            print("行数:", one_hot.size(1))
            if max_value >= one_hot.size(1):
                print("Error: label contains values out of range.")
            else:
                print("All values in label are within the range of one_hot.")
            out_of_range_values = torch.where(label >= one_hot.size(1))
            print("Values in label that are out of range:", out_of_range_values)
            '''
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            # assert label is None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine

        return output


class ccnet(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output\n
    https://ieeexplore.ieee.org/document/9512475
    '''

    def __init__(self, num_classes):
        super(ccnet, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.num_classes = num_classes

        # Competitive blocks with pre-defined kernel sizes, strides, and padding
        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17,
                                                 init_ratio=1, weight=0.8)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8,
                                                 init_ratio=0.5, o2=24, weight=0.8)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3,
                                                 init_ratio=0.25, weight=0.8)

        # Total input size to the fully connected layer (after flattening)
        # 54 channels (9+36+9) and 5x4 feature map size
        self.fc = torch.nn.Linear(13152, 4096)  # Updated to 1080 input features
        self.fc1 = torch.nn.Linear(4096, 2048)

        # Dropout for regularization
        self.drop = torch.nn.Dropout(p=0.5)

        # ArcMarginProduct for angular margin loss
        self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, x, y=None):
        # Forward pass through the Competitive Blocks
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        # Flatten and concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Fully connected layers
        x1 = self.fc(x)
        x = self.fc1(x1)

        # Concatenate the features from both fc layers
        fe = torch.cat((x1, x), dim=1)

        # Apply dropout
        x = self.drop(x)

        # Apply the ArcMarginProduct layer for classification
        x = self.arclayer_(x, y)

        # Return the classification output and normalized feature vector
        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        # Extract feature code without passing through the final classification layer
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        # Flatten the outputs
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)

        # Concatenate the outputs from the three blocks
        x = torch.cat((x1, x2, x3), dim=1)

        # Pass through the fully connected layers
        x = self.fc(x)
        x = self.fc1(x)

        # Normalize the feature vector
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x


class CompetitiveBlock(nn.Module):
    '''
    DESCRIPTION: an implementation of the Competitive Block::

    [CB = LGC + argmax + PPU] \n

    INPUTS: \n

    channel_in: only support 1 \n
    n_competitor: number of channels of the LGC (channel_out)  \n

    ksize, stride, padding: 2D convolution parameters \n

    init_ratio: scale factor of the initial parameters (receptive filed) \n

    o1, o2: numbers of channels of the conv_1 and conv_2 layers in the PPU, respectively. (PPU parameters)
    '''

    def __init__(self, channel_in, n_competitor, ksize, stride, padding, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock, self).__init__()

        assert channel_in == 1

        self.channel_in = 1
        self.n_competitor = n_competitor

        self.init_ratio = init_ratio

        # LGC
        self.gabor_conv2d = GaborConv2d(channel_in=1, channel_out=n_competitor, kernel_size=ksize, stride=stride,
                                        padding=padding, init_ratio=init_ratio)

        # soft-argmax
        self.a = nn.Parameter(torch.FloatTensor([1]))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.argmax = nn.Softmax(dim=1)

        # PPU
        self.conv1 = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor_conv2d(x)

        x = (x - self.b) * self.a

        x = self.argmax(x)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)

        return x


class compnet(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output\n
    https://ieeexplore.ieee.org/document/9512475
    '''

    def __init__(self, num_classes):
        super(compnet, self).__init__()

        self.num_classes = num_classes

        # Competitive blocks with different kernel sizes, strides, and padding
        self.cb1 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=0, init_ratio=1)
        self.cb2 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=17, stride=3, padding=0, init_ratio=0.5)
        self.cb3 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=0, init_ratio=0.25)

        # Fully connected layer
        self.fc = torch.nn.Linear(9708, 512)  # Adjusted for the flattened input size
        self.drop = torch.nn.Dropout(p=0.25)
        self.arclayer = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, x, y=None):
        # Forward pass through the Competitive Blocks
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        # Flatten the outputs of each competitive block
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)

        # Concatenate the outputs from all blocks
        x = torch.cat((x1, x2, x3), dim=1)

        # Pass through the fully connected layer
        x1 = self.fc(x)

        # Apply dropout
        x = self.drop(x1)

        # Use ArcMarginProduct for classification
        output = self.arclayer(x, y)

        # Return both the classification output and the feature vector (before ArcMarginProduct)
        feature_vector = x1  # The feature vector is the output of the fully connected layer
        return output, feature_vector

    def getFeatureCode(self, x):
        # Extract feature code without passing through the final classification layer
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        # Flatten the outputs
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)

        # Concatenate the outputs from the three blocks
        x = torch.cat((x1, x2, x3), dim=1)

        # Pass through the fully connected layer
        x = self.fc(x)

        # Normalize the feature vector
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x


class co3net(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output\n
    https://ieeexplore.ieee.org/document/9512475
    '''

    def __init__(self, num_classes):
        super(co3net, self).__init__()

        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1)
        self.cb2 = CompetitiveBlock(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24)
        self.cb3 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25)

        self.fc = torch.nn.Linear(17328, 4096)  # <---
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        # self.arclayer = torch.nn.Linear(1024,num_classes)
        self.arclayer = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        # g = self.g1(x)

        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = torch.cat((x1, x), dim=1)
        x = self.drop(x)
        x = self.arclayer(x, y)

        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x




