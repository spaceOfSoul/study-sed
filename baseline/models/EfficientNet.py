
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# sigmoid
class Swish(nn.Module):
    def __init__(self, train_beta=False):
        super(Swish, self).__init__()
        if train_beta:
            self.weight = Parameter(torch.Tensor([1.]))
        else:
            self.weight = 1.0

    def forward(self, input):
        return input * torch.sigmoid(self.weight * input)

# squeeze
# fully connected and pooling
class SqeezeExcitation(nn.Module):
    def __init__(self, inplanes, se_ratio):
        super(SqeezeExcitation, self).__init__()
        hidden_dim = int(inplanes*se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(inplanes, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, inplanes, bias=False)
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = x * out.expand_as(x)
        return out

class Bottleneck(nn.Module):
    def __init__(self,inplanes, planes, kernel_size, stride, expand, se_ratio, prob=1.0):
        super(Bottleneck, self).__init__()

        # it has depthwise separable convolution
        if expand == 1:
            self.conv2 = nn.Conv2d(inplanes*expand, inplanes*expand, kernel_size=kernel_size, stride=stride,
                                   padding=kernel_size//2, groups=inplanes*expand, bias=False)
            self.bn2 = nn.BatchNorm2d(inplanes*expand, momentum=0.99, eps=1e-3)
            self.se = SqeezeExcitation(inplanes*expand, se_ratio)
            self.conv3 = nn.Conv2d(inplanes*expand, planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes, momentum=0.99, eps=1e-3)
        else:
            self.conv1 = nn.Conv2d(inplanes, inplanes*expand, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(inplanes*expand, momentum=0.99, eps=1e-3)
            self.conv2 = nn.Conv2d(inplanes*expand, inplanes*expand, kernel_size=kernel_size, stride=stride,
                                   padding=kernel_size//2, groups=inplanes*expand, bias=False) # group is inplanes*expand, it is convolution independent for channel
            self.bn2 = nn.BatchNorm2d(inplanes*expand, momentum=0.99, eps=1e-3)
            self.se = SqeezeExcitation(inplanes*expand, se_ratio) # adjust channel weight
            self.conv3 = nn.Conv2d(inplanes*expand, planes, kernel_size=1, bias=False) # and combine after depthwise separable convolution
            self.bn3 = nn.BatchNorm2d(planes, momentum=0.99, eps=1e-3)

        self.swish = Swish()
        self.correct_dim = (stride == 1) and (inplanes == planes)
        #self.correct_dim = True
        self.prob = torch.Tensor([prob])

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.prob):
                # drop
                return x

        if hasattr(self, 'conv1'):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.swish(out)
        else:
            out = x

        out = self.conv2(out) # depth wise conv
        out = self.bn2(out)
        out = self.swish(out)

        out = self.se(out)


        out = self.conv3(out)
        out = self.bn3(out)

        if self.correct_dim:
            out += x

        return out


class MBConv(nn.Module):
    def __init__(self, inplanes, planes, repeat, kernel_size, stride, expand, se_ratio, sum_layer, count_layer=None, pl=0.5):
        super(MBConv, self).__init__()
        layer = []

        # not drop(stchastic depth)
        layer.append(Bottleneck(inplanes, planes, kernel_size, stride, expand, se_ratio))

        for l in range(1, repeat):
            if count_layer is None:
                layer.append(Bottleneck(planes, planes, kernel_size, 1, expand, se_ratio))
            else:
                # stochastic depth
                prob = 1.0 - (count_layer + l) / sum_layer * (1 - pl)
                layer.append(Bottleneck(planes, planes, kernel_size, 1, expand, se_ratio, prob=prob))

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        out = self.layer(x)
        return out

class Upsample(nn.Module):
    def __init__(self, scale):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)


class Flatten(nn.Module):
    def __init(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class EfficientNet(nn.Module):
    def __init__(self, width_coef=1., depth_coef=1., scale=1.,
                 dropout_ratio=0.5, se_ratio=0.25, stochastic_depth=False, pl=0.5):
        super(EfficientNet, self).__init__()

        channels = [64, 64, 64, 64, 64, 128, 128, 128, 128]
        expands = [2, 2, 2, 1, 1, 1, 1]
        repeats = [1, 1, 1, 1, 1, 1, 1]
        strides = [1, 1, 1, 1, 1, 1, 1]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 3]
        depth = depth_coef
        width = width_coef

        channels = [round(x * width) for x in channels]
        repeats = [round(x * depth) for x in repeats]
        sum_layer = sum(repeats)

        self.upsample = Upsample(scale)
        self.swish = Swish()

        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3)
        ))

        for i in range(7):
            layers.append(MBConv(
                inplanes=channels[i],
                planes=channels[i+1],
                repeat=repeats[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                expand=expands[i],
                se_ratio=se_ratio,
                sum_layer=sum_layer,
                count_layer=sum(repeats[:i]),
                pl=pl
            ))

        layers.append(nn.Sequential(
            nn.Conv2d(channels[6], channels[7], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[7], momentum=0.99, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d((157, 1)),
            nn.Dropout(p=dropout_ratio)
        ))

        self.cnn = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.cnn.state_dict(), filename)

    def forward(self, x):
        x = self.upsample(x)
        x = self.cnn(x)
        return x

class EfficientNetWrapper(nn.Module):
    def __init__(self, efficient_net):
        super(EfficientNetWrapper, self).__init__()
        self.efficient_net = efficient_net

    def forward(self, x):
        return self.efficient_net(x)

    def load_state_dict(self, state_dict, strict=True):
        self.efficient_net.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.efficient_net.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.efficient_net.state_dict(), filename)


if __name__ == "__main__":
    model = EfficientNet()
    for name, layer in model.named_modules():
        print(f"Layer name: {name}, Layer: {layer}")

    input_data = torch.randn(1, 1, 224, 224)
    output = model(input_data)
    print(f"Output shape: {output.shape}")

    # 모델의 state_dict 저장
    torch.save(model.state_dict(), 'efficientnet_state_dict.pth')

    # 새로운 모델 인스턴스화 및 state_dict 로드
    new_model = EfficientNet()
    new_model.load_state_dict(torch.load('efficientnet_state_dict.pth'))

    # 새로운 모델의 출력을 확인하여 동일한지 확인
    new_output = new_model(input_data)
    print(f"New output shape: {new_output.shape}")
    print(f"Outputs are the same: {torch.allclose(output, new_output)}")