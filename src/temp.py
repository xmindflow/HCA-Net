import _init_paths
from pose_code.atthourglass import Bottleneck, HourglassNet, Hourglass
import torch
from torchsummary import summary


m = HourglassNet(
    Bottleneck,
    num_stacks=1,
    num_blocks=1,
    num_classes=11
)

x = torch.rand((8, 3, 256, 256))

# summary(m, x)

y = m(x)  

# print(y[0].shape)
