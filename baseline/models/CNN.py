import torch.nn as nn
import torch
from torchvision import models

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNN(nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)], poolingFunc = "avg", skip_connection = True
                 ):
        super(CNN, self).__init__()
        self.nb_filters = nb_filters
        cnn = nn.Sequential()
        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
            if dropout is not None:
                cnn.add_module('dropout{0}'.format(i),
                               nn.Dropout(dropout))

        batch_norm = True
        # 128x862x64
        #for i in range(len(nb_filters)):
        #    conv(i, batch_norm, conv_dropout, activ=activation)
        #    # bs x tframe x mels
        #    if poolingFunc == "avg":
        #        cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))
        #    elif  poolingFunc == "max":
        #        cnn.add_module('pooling{0}'.format(i), nn.MaxPool2d(pooling[i]))

        # using skip connection (with rasidual block)
        in_channels = n_in_channel

        for i in range(len(nb_filters)):
            out_channels = nb_filters[i]
            cnn.add_module(f'resblock{i}', ResidualConvBlock(in_channels, out_channels, kernel_size[i], stride[i], padding[i],
                                                            activation, conv_dropout))
            
            in_channels = out_channels  # 다음 블록의 입력 채널을 현재의 출력 채널로 설정

        for i in range(len(pooling)):
            cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))

        self.cnn = cnn

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.cnn.state_dict(), filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='relu', dropout=None):
        super(ResidualConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)

        # Residual connection을 위한 downsample 레이어가 필요한 경우
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
            )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)
        return out

class Resnet(nn.Module):
    def __init__(self, n_in_channel, activation="relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)], resnet_model='resnet50'):
        super(Resnet, self).__init__()
        self.n_in_channel = n_in_channel
        self.nb_filters = nb_filters

        if resnet_model == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
        elif resnet_model == 'resnet34':
            self.base_model = models.resnet34(pretrained=True)
        elif resnet_model == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
        else:
            raise ValueError('Unsupported ResNet model')

        # resnet input channel is 3. Need change channel
        if n_in_channel != 3:
            self.base_model.conv1 = nn.Conv2d(n_in_channel, nb_filters[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0], bias=False)

        self.cnn = nn.Sequential(*list(self.base_model.children())[:-2])
        self.cnn.add_module("adjust_conv2d",nn.Conv2d(2048, 128, 1))
        # pollings
        self.pool = nn.Sequential()
        for i in range(len(pooling)):
            self.pool.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))

        # dropout
        if conv_dropout > 0:
            self.dropout = nn.Dropout(conv_dropout)
    
    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def forward(self, x):
        print(f'before cnn {x.shape}')
        x = self.cnn(x)
        print(f'before pooling {x.shape}')
        x = self.pool(x)
        #for _, module in self.pool.named_children():
        #    # size check
        #    if x.size(2) >= module.kernel_size[0] and x.size(3) >= module.kernel_size[1]:
        #        x = module(x)
        #    else:
        #        continue
        print(f'after pooling {x.shape}')

        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x