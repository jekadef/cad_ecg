import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class EcgNetKKK(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockKKK(in_channels, conv_channels)
        self.block2 = EcgBlockKKK(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockKKK(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockKKK(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockKKK(conv_channels * 6, conv_channels * 8)
        self.block6 = EcgBlockKKK(conv_channels * 8, conv_channels * 10)
        self.block7 = EcgBlockKKK(conv_channels * 10, conv_channels * 12)
        self.linear1 = nn.Linear(12288, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        block_out = self.block6(block_out)
        block_out = self.block7(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockKKK(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1,5))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetJJJ(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockJJJ(in_channels, conv_channels)
        self.block2 = EcgBlockJJJ(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockJJJ(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockJJJ(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockJJJ(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(12288, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockJJJ(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1,5))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetIII(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockIII(in_channels, conv_channels)
        self.block2 = EcgBlockIII(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockIII(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(12288, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockIII(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1,5))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)








class EcgNetHHH(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockHHH(in_channels, conv_channels)
        self.block2 = EcgBlockHHH(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockHHH(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockHHH(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockHHH(conv_channels * 6, conv_channels * 8)
        self.block6 = EcgBlockHHH(conv_channels * 8, conv_channels * 10)
        self.block7 = EcgBlockHHH(conv_channels * 10, conv_channels * 12)
        self.linear1 = nn.Linear(12288, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        block_out = self.block6(block_out)
        block_out = self.block7(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockHHH(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1,3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetGGG(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockGGG(in_channels, conv_channels)
        self.block2 = EcgBlockGGG(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockGGG(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockGGG(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockGGG(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(12288, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockGGG(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1,3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetFFF(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockFFF(in_channels, conv_channels)
        self.block2 = EcgBlockFFF(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockFFF(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(12288, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockFFF(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1,3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetEEE(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockEEE(in_channels, conv_channels)
        self.block2 = EcgBlockEEE(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockEEE(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockEEE(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockEEE(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(39936, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockEEE(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), stride=1, kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)

class EcgNetDDD(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockDDD(in_channels, conv_channels)
        self.block2 = EcgBlockDDD(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockDDD(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockDDD(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockDDD(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(5120, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockDDD(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), stride=1, kernel_size=7, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)

class EcgNetCCC(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockCCC(in_channels, conv_channels)
        self.block2 = EcgBlockCCC(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockCCC(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockCCC(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockCCC(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(5120, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockCCC(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), stride=1, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)

class EcgNetBBB(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockBBB(in_channels, conv_channels)
        self.block2 = EcgBlockBBB(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockBBB(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockBBB(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockBBB(conv_channels * 6, conv_channels * 8)
        self.block6 = EcgBlockBBB(conv_channels * 8, conv_channels * 10)
        self.linear1 = nn.Linear(1920, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        block_out = self.block6(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockBBB(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), stride=1, kernel_size=7, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)

class EcgNetAAA(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockAAA(in_channels, conv_channels)
        self.block2 = EcgBlockAAA(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockAAA(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockAAA(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockAAA(conv_channels * 6, conv_channels * 8)
        self.block6 = EcgBlockAAA(conv_channels * 8, conv_channels * 10)
        self.linear1 = nn.Linear(1920, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        block_out = self.block6(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockAAA(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), stride=1, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)

class EcgNetZZ(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockZZ(in_channels, conv_channels)
        self.block2 = EcgBlockZZ(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockZZ(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockZZ(conv_channels * 4, conv_channels * 6)
        self.linear1 = nn.Linear(2304, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockZZ(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=7, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)

class EcgNetYY(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm3d(num_features=in_channels)
        self.block1 = EcgBlockYY(in_channels, conv_channels)
        self.block2 = EcgBlockYY(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockYY(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(2048, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockYY(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, int(conv_channels), kernel_size=7, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)

class EcgNetUU(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm3d(num_features=in_channels)
        self.block1 = EcgBlockUU(in_channels, conv_channels)
        self.linear1 = nn.Linear(36864, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockUU(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)


class EcgNetVV(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm3d(num_features=in_channels)
        self.block1 = EcgBlockVV(in_channels, conv_channels)
        self.linear1 = nn.Linear(36864, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockVV(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)


class EcgNetWW(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm3d(num_features=in_channels)
        self.block1 = EcgBlockWW(in_channels, conv_channels)
        self.block2 = EcgBlockWW(conv_channels, 32)
        self.linear1 = nn.Linear(16384, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockWW(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)


class EcgNetXX(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm3d(num_features=in_channels)
        self.block1 = EcgBlockXX(in_channels, conv_channels)
        self.block2 = EcgBlockXX(conv_channels, 32)
        self.linear1 = nn.Linear(16384, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockXX(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)

class EcgNetSS(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockSS(in_channels, conv_channels)
        self.linear1 = nn.Linear(36864, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockSS(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)

class EcgNetTT(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockTT(in_channels, conv_channels)
        self.linear1 = nn.Linear(36864, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockTT(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)

class EcgNetRR(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockRR(in_channels, conv_channels)
        self.block2 = EcgBlockRR(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockRR(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockRR(conv_channels * 4, conv_channels * 6)
        self.linear1 = nn.Linear(3072, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockRR(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)

class EcgNetQQ(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockQQ(in_channels, conv_channels)
        self.block2 = EcgBlockQQ(conv_channels, conv_channels * 2)
        self.linear1 = nn.Linear(16384, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockQQ(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)

class EcgNetPP(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockPP(in_channels, conv_channels)
        self.block2 = EcgBlockPP(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockPP(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockPP(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockPP(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(10240, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)

class EcgBlockPP(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 10), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetOO(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockOO(in_channels, conv_channels)
        self.block2 = EcgBlockOO(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockOO(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockOO(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockOO(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(10240, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockOO(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 5), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetNN(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockNN(in_channels, conv_channels)
        self.block2 = EcgBlockNN(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockNN(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockNN(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockNN(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(10240, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockNN(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 3), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetMM(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockMM(in_channels, conv_channels)
        self.block2 = EcgBlockMM(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockMM(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(47360, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockMM(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 10), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetLL(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockLL(in_channels, conv_channels)
        self.block2 = EcgBlockLL(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockLL(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(47360, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockLL(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 5), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetKK(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockKK(in_channels, conv_channels)
        self.block2 = EcgBlockKK(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockKK(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(47360, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockKK(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 3), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetEE(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockEE(in_channels, conv_channels)
        self.block2 = EcgBlockEE(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockEE(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockEE(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockEE(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(5120, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockEE(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), stride=1, kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetFF(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockFF(in_channels, conv_channels)
        self.block2 = EcgBlockFF(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockFF(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockFF(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockFF(conv_channels * 6, conv_channels * 8)
        self.block6 = EcgBlockFF(conv_channels * 8, conv_channels * 10)
        self.linear1 = nn.Linear(1920, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        block_out = self.block6(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockFF(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), stride=1, kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetII(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockII(in_channels, conv_channels)
        self.block2 = EcgBlockII(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockII(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockII(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockII(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(5120, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockII(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetJJ(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockJJ(in_channels, conv_channels)
        self.block2 = EcgBlockJJ(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockJJ(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockJJ(conv_channels * 4, conv_channels * 6)
        self.block5 = EcgBlockJJ(conv_channels * 6, conv_channels * 8)
        self.linear1 = nn.Linear(5120, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockJJ(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=10, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetHH(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockHH(in_channels, conv_channels)
        self.block2 = EcgBlockHH(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockHH(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(480, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockHH(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=30, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetGG(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockGG(in_channels, conv_channels)
        self.block2 = EcgBlockGG(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockGG(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(1280, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockGG(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 30), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((1, 10))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetCC(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm3d(num_features=in_channels)
        self.block1 = EcgBlockCC(in_channels, conv_channels)
        self.block2 = EcgBlockCC(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockCC(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(2048, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockCC(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)


class EcgNetDD(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm3d(num_features=in_channels)
        self.block1 = EcgBlockDD(in_channels, conv_channels)
        self.block2 = EcgBlockDD(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockDD(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(2048, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockDD(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)


class EcgNetAA(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockAA(in_channels, conv_channels)
        self.block2 = EcgBlockAA(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockAA(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(6144, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockAA(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)


class EcgNetBB(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockBB(in_channels, conv_channels)
        self.block2 = EcgBlockBB(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockBB(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockBB(conv_channels * 4, conv_channels * 6)
        self.linear1 = nn.Linear(2304, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockBB(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)


class EcgNetX(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockX(in_channels, conv_channels)
        self.block2 = EcgBlockX(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockX(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(480, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockX(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=100, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetY(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockY(in_channels, conv_channels)
        self.block2 = EcgBlockY(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockY(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(480, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockY(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=50, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetV(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockV(in_channels, conv_channels)
        self.block2 = EcgBlockV(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockV(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(1280, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockV(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 100), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((1, 10))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetW(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockW(in_channels, conv_channels)
        self.block2 = EcgBlockW(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockW(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(1280, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockW(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=(8, 50), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((1, 10))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetT(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockT(in_channels, conv_channels)
        self.block2 = EcgBlockT(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockT(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(23680, 4096)
        self.linear2 = nn.Linear(4096, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        linear_output = self.linear2(linear_output)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockT(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetU(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockU(in_channels, conv_channels)
        self.block2 = EcgBlockU(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockU(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockU(conv_channels * 4, conv_channels * 6)
        self.linear1 = nn.Linear(11712, 1024)
        self.linear2 = nn.Linear(1024, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        linear_output = self.linear2(linear_output)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockU(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetS(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockS(in_channels, conv_channels)
        self.block2 = EcgBlockS(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockS(conv_channels * 2, conv_channels * 4)
        self.block4 = EcgBlockS(conv_channels * 4, conv_channels * 6)
        self.linear1 = nn.Linear(11712, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockS(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetR(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockR(in_channels, conv_channels)
        self.block2 = EcgBlockR(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockR(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(23680, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockR(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetQ(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockQ(in_channels, conv_channels)
        self.block2 = EcgBlockQ(conv_channels, conv_channels * 2)
        self.block3 = EcgBlockQ(conv_channels * 2, conv_channels * 4)
        self.linear1 = nn.Linear(6144, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockQ(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        return self.max_pool(block_out)


class EcgNetP(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockP(in_channels, conv_channels)
        self.block2 = EcgBlockP(conv_channels, conv_channels * 2)
        self.linear1 = nn.Linear(12288, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockP(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        return self.max_pool(block_out)


class EcgNetO(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockO(in_channels, conv_channels)
        self.block2 = EcgBlockO(conv_channels, conv_channels * 4)
        self.block3 = EcgBlockO(conv_channels * 4, conv_channels * 16)
        self.linear1 = nn.Linear(384 * 5, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockO(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels/2), kernel_size=100, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=10)
        self.conv2 = nn.Conv1d(int(conv_channels/2), conv_channels, kernel_size=100, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetN(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.block1 = EcgBlockN(in_channels, conv_channels)
        self.block2 = EcgBlockN(conv_channels, conv_channels * 4)
        self.linear1 = nn.Linear(96 * 50, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockN(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels/2), kernel_size=100, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(int(conv_channels/2), conv_channels, kernel_size=100, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetM(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockM(in_channels, conv_channels)
        self.block2 = EcgBlockM(conv_channels, conv_channels * 4)
        self.block3 = EcgBlockM(conv_channels * 4, conv_channels * 16)
        self.linear1 = nn.Linear(128 * 8 * 5, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockM(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels/2), kernel_size=(8, 100), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(conv_channels/2), conv_channels, kernel_size=(8, 100), padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((1, 10))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetL(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockL(in_channels, conv_channels)
        self.block2 = EcgBlockL(conv_channels, conv_channels * 4)
        self.linear1 = nn.Linear(32 * 8 * 50, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockL(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels/2), kernel_size=(8, 100), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(conv_channels/2), conv_channels, kernel_size=(8, 100), padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((1, 10))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetK(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.block1 = EcgBlockK(in_channels, conv_channels)
        self.block2 = EcgBlockK(conv_channels, conv_channels * 4)
        self.block3 = EcgBlockK(conv_channels * 4, conv_channels * 16)
        self.linear1 = nn.Linear(512 * 171, 16384) ### change values
        self.linear2 = nn.Linear(16384, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.linear1(conv_flat)
        linear_output = self.linear2(linear_output)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockK(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels/2), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(conv_channels/2), conv_channels, kernel_size=5, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetJ(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)

        self.block1 = EcgBlockJ(in_channels, conv_channels)
        self.block2 = EcgBlockJ(conv_channels, conv_channels * 4)
        # self.block3 = EcgBlockJ(conv_channels, conv_channels * 16)

        self.linear1 = nn.Linear(512 * 19 * 7, 4256) ### change values
        self.linear2 = nn.Linear(4256, 2)
        # self.linear1 = nn.Linear(512 * 19 * 7, 10944) ### change values
        # self.linear2 = nn.Linear(10944, 1824)
        # self.linear3 = nn.Linear(1824, 456)
        # self.linear4 = nn.Linear(456, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv1d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        # block_out = self.block3(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.linear1(conv_flat)
        linear_output = self.linear2(linear_output)
        # linear_output = self.linear3(linear_output)
        # linear_output = self.linear4(linear_output)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockJ(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels/2), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(conv_channels/2), conv_channels, kernel_size=5, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetI(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)

        self.block1 = EcgBlockI(in_channels, conv_channels)
        self.block2 = EcgBlockI(conv_channels, conv_channels * 4)
        self.block3 = EcgBlockI(conv_channels * 4, conv_channels * 16)

        self.linear1 = nn.Linear(94720, 4256) ### change values
        self.linear2 = nn.Linear(4256, 2)

        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv1d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.linear1(conv_flat)
        linear_output = self.linear2(linear_output)

        return linear_output, self.head_softmax(linear_output)


class EcgBlockI(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels/2), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(int(conv_channels/2), conv_channels, kernel_size=5, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetH(nn.Module):
    def __init__(self, in_channels=8, conv_channels=32):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)

        self.block1 = EcgBlockH(in_channels, conv_channels)
        self.block2 = EcgBlockH(conv_channels, conv_channels * 4)

        self.linear1 = nn.Linear(128 * 555, 2)  ### change values

        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv1d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.linear1(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockH(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels/2), kernel_size=5, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(int(conv_channels/2), conv_channels, kernel_size=5, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetG(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)

        self.block1 = EcgBlockG(in_channels, conv_channels)
        self.block2 = EcgBlockG(conv_channels, conv_channels * 4)
        self.block3 = EcgBlockG(conv_channels * 4, conv_channels * 16)

        self.head_linear = nn.Linear(384 * 5, 2) ### change values
        # self.head_linear = nn.Linear(6400, 1600)
        # self.head_linear = nn.Linear(1600, 400)
        # self.head_linear = nn.Linear(400, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv1d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                 )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockG(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels/2), kernel_size=10, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(int(conv_channels/2), conv_channels, kernel_size=10, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetF(nn.Module):
    def __init__(self, in_channels=8, conv_channels=24):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm1d(num_features=in_channels)

        self.block1 = EcgBlockF(in_channels, conv_channels)
        self.block2 = EcgBlockF(conv_channels, conv_channels * 4)

        self.head_linear = nn.Linear(96 * 50, 2) ### change values

        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv1d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockF(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, int(conv_channels/2), kernel_size=10, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(int(conv_channels/2), conv_channels, kernel_size=10, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetE(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)

        self.block1 = EcgBlockE(in_channels, conv_channels)
        self.block2 = EcgBlockE(conv_channels, conv_channels * 4)
        self.block3 = EcgBlockE(conv_channels * 4, conv_channels * 16)

        self.head_linear = nn.Linear(128 * 8 * 5, 2)
        # self.head_linear = nn.Linear(6400, 1600)
        # self.head_linear = nn.Linear(1600, 400)
        # self.head_linear = nn.Linear(400, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv2d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockE(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels/2), kernel_size=(8, 10), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(conv_channels/2), conv_channels, kernel_size=(8, 10), padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((1, 10))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNetD(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)

        self.block1 = EcgBlockD(in_channels, conv_channels)
        self.block2 = EcgBlockD(conv_channels, conv_channels * 4)

        self.head_linear = nn.Linear(32 * 8 * 50, 2)
        # self.head_linear = nn.Linear(6400, 1600)
        # self.head_linear = nn.Linear(1600, 400)
        # self.head_linear = nn.Linear(400, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv2d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlockD(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels/2), kernel_size=(8, 10), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(conv_channels/2), conv_channels, kernel_size=(8, 10), padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((1, 10))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)


class EcgNet(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batch_norm = nn.BatchNorm2d(num_features=in_channels)

        self.block1 = EcgBlock(in_channels, conv_channels)
        self.block2 = EcgBlock(conv_channels, conv_channels * 4)
        self.head_linear = nn.Linear(32 * 8 * 50, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv2d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batch_norm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)

        #conv_flat = block_out.view(-1, 32*32*2)
        conv_flat = block_out.view(block_out.size(0), -1)

        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class EcgBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, int(conv_channels/2), kernel_size=(1, 10), padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        # max pool
        self.conv2 = nn.Conv2d(int(conv_channels/2), conv_channels, kernel_size=(1, 10), padding='same')
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((1, 10))

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.max_pool(block_out)
