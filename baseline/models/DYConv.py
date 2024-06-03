import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=True, groups=1):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) # it is allow unity value
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.groups = groups
        
        if padding == 'same':
            self.padding_size = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        else:
            self.padding_size = padding
        
        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, self.padding_size, bias=bias, groups=groups)
        
        # freq adapt layer
        self.freq_adapt = nn.Parameter(torch.randn(1, out_channels, 1, 1)) 

    def forward(self, x):
        x = self.conv(x)
        
        batch_size, channels, height, width = x.size()
        
        # filter size adjust (repeat for input shape. )
        freq_adapted_filters = self.freq_adapt.repeat(batch_size, 1, height, width)
        x = x * freq_adapted_filters # applt filter
        
        return x

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias
    