import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt

from functools import reduce
from operator import __add__
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)

class APN(nn.Module):
    def __init__(self):
        super(APN , self).__init__()
        self.conv_1 = Conv2dSamePadding(1,1,5, bias=False)       
        self.conv_2 = Conv2dSamePadding(1,1,3, bias=False)
        self.conv_3 = Conv2dSamePadding(1,16,5, bias=False)
        self.conv_4 = Conv2dSamePadding(16,16,5, bias=False)
        self.conv_5 = Conv2dSamePadding(16,16,5, bias=False)       
        self.conv_6 = Conv2dSamePadding(16,1,3, bias=False)
        self.tanh = nn.Tanh()
    def forward(self, x):
        # size = x.size()
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.conv_4(x3)
        x5 = self.conv_5(x4)
        x6 = self.conv_6(x5)
        x6 = self.tanh(x6)
        x7 = torch.cat((x6, x6, x6))
        
        return x7



unsharp_mask_model = APN()  # 實例化整個網絡，init 中可設置各種參数
print(unsharp_mask_model)

img = Image.open('/Users/chenshirley/Downloads/cag.png')
img = TF.to_tensor(img)
img = img[0,:,:]
img = img.unsqueeze(0)
result = unsharp_mask_model(img)