import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
        super(Attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = max(in_planes // 4, 4)

        if pool_dim != 'both':
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
            self._initialize_weights()
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=3)
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=2)
        elif self.pool_dim == 'both':
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == 'chan':
            x = torch.mean(x, dim=1)

        if self.pool_dim != 'both':
            x = self.conv1d1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

        return F.softmax(x / self.temperature, dim=1)

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4,
                 temperature=31, pool_dim='freq', groups=1):
        super(DynamicConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim
        self.groups = groups
        self.n_basis_kernels = n_basis_kernels

        self.attention = Attention2d(in_channels, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim)
        self.weight = nn.Parameter(torch.randn(n_basis_kernels, out_channels, in_channels // groups, kernel_size, kernel_size), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_channels))
        else:
            self.bias = None

        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x):
        softmax_attention = self._get_attention(x)
        batch_size = x.size(0)
        aggregate_weight = self.weight.view(-1, self.in_channels // self.groups, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)

        output = output.view(batch_size, self.n_basis_kernels, self.out_channels, output.size(-2), output.size(-1))
        output = torch.sum(output * softmax_attention, dim=1)

        return output

    def _get_attention(self, x):
        if self.pool_dim in ['freq', 'chan']:
            return self.attention(x).unsqueeze(2).unsqueeze(4)
        elif self.pool_dim == 'time':
            return self.attention(x).unsqueeze(2).unsqueeze(3)
        elif self.pool_dim == 'both':
            return self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)