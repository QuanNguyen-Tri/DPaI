import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    
    def __init__(self):
        super(Residual, self).__init__()
    
    def forward(self, x, y):
        return x + y
    

class SparseSpeedupBench(object):
    """Class to benchmark speedups for convolutional layers.

    Basic usage:
    1. Assing a single SparseSpeedupBench instance to class (and sub-classes with conv layers).
    2. Instead of forwarding input through normal convolutional layers, we pass them through the bench:
        self.bench = SparseSpeedupBench()
        self.conv_layer1 = nn.Conv2(3, 96, 3)

        if self.bench is not None:
            outputs = self.bench.forward(self.conv_layer1, inputs, layer_id='conv_layer1')
        else:
            outputs = self.conv_layer1(inputs)
    3. Speedups of the convolutional layer will be aggregated and print every 1000 mini-batches.
    """
    def __init__(self):
        self.layer_timings = {}
        self.layer_timings_channel_sparse = {}
        self.layer_timings_sparse = {}
        self.iter_idx = 0
        self.layer_0_idx = None
        self.total_timings = []
        self.total_timings_channel_sparse = []
        self.total_timings_sparse = []

    def get_density(self, x):
        return (x.data!=0.0).sum().item()/x.numel()

    def print_weights(self, w, layer):
        # w dims: out, in, k1, k2
        #outers = []
        #for outer in range(w.shape[0]):
        #    inners = []
        #    for inner in range(w.shape[1]):
        #        n = np.prod(w.shape[2:])
        #        density = (w[outer, inner, :, :] != 0.0).sum().item() / n
        #        #print(density, w[outer, inner])
        #        inners.append(density)
        #    outers.append([np.mean(inners), np.std(inner)])
        #print(outers)
        #print(w.shape, (w!=0.0).sum().item()/w.numel())
        pass

    def forward(self, layer, x, layer_id):
        if self.layer_0_idx is None: self.layer_0_idx = layer_id
        if layer_id == self.layer_0_idx: self.iter_idx += 1
        self.print_weights(layer.weight.data, layer)

        # calc input sparsity
        sparse_channels_in = ((x.data != 0.0).sum([2, 3]) == 0.0).sum().item()
        num_channels_in = x.shape[1]
        batch_size = x.shape[0]
        channel_sparsity_input = sparse_channels_in/float(num_channels_in*batch_size)
        input_sparsity = self.get_density(x)

        # bench dense layer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = layer(x)
        end.record()
        start.synchronize()
        end.synchronize()
        time_taken_s = start.elapsed_time(end)/1000.0

        # calc weight sparsity
        num_channels = layer.weight.shape[1]
        sparse_channels = ((layer.weight.data != 0.0).sum([0, 2, 3]) == 0.0).sum().item()
        channel_sparsity_weight = sparse_channels/float(num_channels)
        weight_sparsity = self.get_density(layer.weight)

        # store sparse and dense timings
        if layer_id not in self.layer_timings:
            self.layer_timings[layer_id] = []
            self.layer_timings_channel_sparse[layer_id] = []
            self.layer_timings_sparse[layer_id] = []
        self.layer_timings[layer_id].append(time_taken_s)
        self.layer_timings_channel_sparse[layer_id].append(time_taken_s*(1.0-channel_sparsity_weight)*(1.0-channel_sparsity_input))
        self.layer_timings_sparse[layer_id].append(time_taken_s*input_sparsity*weight_sparsity)

        if self.iter_idx % 1000 == 0:
            self.print_layer_timings()
            self.iter_idx += 1

        return x

    def print_layer_timings(self):
        total_time_dense = 0.0
        total_time_sparse = 0.0
        total_time_channel_sparse = 0.0
        print('\n')
        for layer_id in self.layer_timings:
            t_dense = np.mean(self.layer_timings[layer_id])
            t_channel_sparse = np.mean(self.layer_timings_channel_sparse[layer_id])
            t_sparse = np.mean(self.layer_timings_sparse[layer_id])
            total_time_dense += t_dense
            total_time_sparse += t_sparse
            total_time_channel_sparse += t_channel_sparse

            print('Layer {0}: Dense {1:.6f} Channel Sparse {2:.6f} vs Full Sparse {3:.6f}'.format(layer_id, t_dense, t_channel_sparse, t_sparse))
        self.total_timings.append(total_time_dense)
        self.total_timings_sparse.append(total_time_sparse)
        self.total_timings_channel_sparse.append(total_time_channel_sparse)

        print('Speedups for this segment:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense, total_time_channel_sparse, total_time_dense/total_time_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense, total_time_sparse, total_time_dense/total_time_sparse))
        print('\n')

        total_dense = np.sum(self.total_timings)
        total_sparse = np.sum(self.total_timings_sparse)
        total_channel_sparse = np.sum(self.total_timings_channel_sparse)
        print('Speedups for entire training:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_channel_sparse, total_dense/total_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_sparse, total_dense/total_sparse))
        print('\n')

        # clear timings
        for layer_id in list(self.layer_timings.keys()):
            self.layer_timings.pop(layer_id)
            self.layer_timings_channel_sparse.pop(layer_id)
            self.layer_timings_sparse.pop(layer_id)



class AlexNet(nn.Module):
    """AlexNet with batch normalization and without pooling.

    This is an adapted version of AlexNet as taken from
    SNIP: Single-shot Network Pruning based on Connection Sensitivity,
    https://arxiv.org/abs/1810.02340

    There are two different version of AlexNet:
    AlexNet-s (small): Has hidden layers with size 1024
    AlexNet-b (big):   Has hidden layers with size 2048

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config='s', num_classes=1000, save_features=False, bench_model=False):
        super(AlexNet, self).__init__()
        self.save_features = save_features
        self.feats = []
        self.densities = []
        self.bench = None if not bench_model else SparseSpeedupBench()

        factor = 1 if config=='s' else 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 1024*factor),
            nn.BatchNorm1d(1024*factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024*factor, 1024*factor),
            nn.BatchNorm1d(1024*factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024*factor, num_classes),
        )

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            if self.bench is not None and isinstance(layer, nn.Conv2d):
                x = self.bench.forward(layer, x, layer_id)
            else:
                x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                if isinstance(layer, nn.Conv2d):
                    self.densities.append((layer.weight.data != 0.0).sum().item()/layer.weight.numel())

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # return F.log_softmax(x, dim=1)
        return x

class LeNet_300_100(nn.Module):
    """Simple NN with hidden layers [300, 100]

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """
    def __init__(self, save_features=None, bench_model=False):
        super(LeNet_300_100, self).__init__()
        self.fc1 = nn.Linear(28*28, 300, bias=True)
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.mask = None

    def forward(self, x):
        x0 = x.view(-1, 28*28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        # return F.log_softmax(x3, dim=1)
        return x3

class MLP_CIFAR10(nn.Module):
    def __init__(self, save_features=None, bench_model=False):
        super(MLP_CIFAR10, self).__init__()

        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3*32*32)))
        x1 = F.relu(self.fc2(x0))
        # return F.log_softmax(self.fc3(x1), dim=1)
        return self.fc3(x1)


class LeNet_5_Caffe(nn.Module):
    """LeNet-5 without padding in the first layer.
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, save_features=None, bench_model=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, bias=True)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=True)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
        # x = F.log_softmax(self.fc4(x), dim=1)

        return x


VGG_CONFIGS = {
    # M for MaxPool, Number for channels
    'like': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'C': [
        64, 64, 'M', 128, 128, 'M', 256, 256, (1, 256), 'M', 512, 512, (1, 512), 'M',
        512, 512, (1, 512), 'M' # tuples indicate (kernel size, output channels)
    ]
}


class VGG16(nn.Module):
    """
    This is a base class to generate three VGG variants used in SNIP paper:
        1. VGG-C (16 layers)
        2. VGG-D (16 layers)
        3. VGG-like

    Some of the differences:
        * Reduced size of FC layers to 512
        * Adjusted flattening to match CIFAR-10 shapes
        * Replaced dropout layers with BatchNorm

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config, num_classes=10, save_features=False, bench_model=False):
        super().__init__()

        self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True)
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = None if not bench_model else SparseSpeedupBench()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()
        if config == 'C' or config == 'D':
            self.classifier = nn.Sequential(
                nn.Linear((512 if config == 'D' else 2048), 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, num_classes),
            )

    @staticmethod
    def make_layers(config, batch_norm=False):
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                kernel_size = 3
                if isinstance(v, tuple):
                    kernel_size, v = v
                conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            if self.bench is not None and isinstance(layer, nn.Conv2d):
                x = self.bench.forward(layer, x, layer_id)
            else:
                x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                    self.densities.append((x.data != 0.0).sum().item()/x.numel())

        x = self.flatten(x)
        x = self.classifier(x)
        # x = self.log_softmax(x)
        return x

class VGG(nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    def __init__(self, plan, conv, num_classes=10, dense_classifier=False):
        super(VGG, self).__init__()
        layer_list = []
        filters = 3
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()
        self.avg_pool2d = nn.AvgPool2d(2)

        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layer_list)        

        self.fc = nn.Linear(512, num_classes)
        if dense_classifier:
            self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = self.avg_pool2d(x)
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.log_softmax(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def vgg_plan(num):
    if num == 11:
        plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 13:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 16:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    elif num == 19:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    else:
        raise ValueError('Unknown VGG model: {}'.format(num))
    return plan

def _vgg(arch, plan, conv, num_classes, dense_classifier, pretrained):
    model = VGG(plan, conv, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

class ConvModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class ConvBNModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvBNModule, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
def vgg11(num_classes, dense_classifier=False, pretrained=False):
    plan = vgg_plan(11)
    return _vgg('vgg11_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg11_bn(num_classes, dense_classifier=False, pretrained=False):
    plan = vgg_plan(11)
    return _vgg('vgg11_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg13(num_classes, dense_classifier=False, pretrained=False):
    plan = vgg_plan(13)
    return _vgg('vgg13_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg13_bn(num_classes, dense_classifier=False, pretrained=False):
    plan = vgg_plan(13)
    return _vgg('vgg13_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg16(num_classes, dense_classifier=False, pretrained=False):
    plan = vgg_plan(16)
    return _vgg('vgg16_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg16_bn(num_classes, dense_classifier=False, pretrained=False):
    plan = vgg_plan(16)
    return _vgg('vgg16_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg19(num_classes, dense_classifier=False, pretrained=False):
    plan = vgg_plan(19)
    return _vgg('vgg19_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg19_bn(num_classes, dense_classifier=False, pretrained=False):
    plan = vgg_plan(19)
    return _vgg('vgg19_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


class WideResNet(nn.Module):
    """Wide Residual Network with varying depth and width.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """
    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.3, save_features=False, bench_model=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bench = None if not bench_model else SparseSpeedupBench()
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, save_features=save_features, bench=self.bench)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, save_features=save_features, bench=self.bench)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, save_features=save_features, bench=self.bench)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(8)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.feats = []
        self.densities = []
        self.save_features = save_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.bench is not None:
            out = self.bench.forward(self.conv1, x, 'conv1')
        else:
            out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        if self.save_features:
            # this is a mess, but I do not have time to refactor it now
            self.feats += self.block1.feats
            self.densities += self.block1.densities
            del self.block1.feats[:]
            del self.block1.densities[:]
            self.feats += self.block2.feats
            self.densities += self.block2.densities
            del self.block2.feats[:]
            del self.block2.densities[:]
            self.feats += self.block3.feats
            self.densities += self.block3.densities
            del self.block3.feats[:]
            del self.block3.densities[:]

        out = self.relu(self.bn1(out))
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        # return self.log_softmax(out)
        return out


class BasicBlock(nn.Module):
    """Wide Residual Network basic block

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, save_features=False, bench=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.res = Residual()
        self.dropout = nn.Dropout(self.droprate)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench
        self.in_planes = in_planes

    def forward(self, x):
        conv_layers = []
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            if self.save_features:
                self.feats.append(x.clone().detach())
                self.densities.append((x.data != 0.0).sum().item()/x.numel())
        else:
            out = self.relu1(self.bn1(x))
            if self.save_features:
                self.feats.append(out.clone().detach())
                self.densities.append((out.data != 0.0).sum().item()/out.numel())
        if self.bench:
            out0 = self.bench.forward(self.conv1, (out if self.equalInOut else x), str(self.in_planes) + '.conv1')
        else:
            out0 = self.conv1(out if self.equalInOut else x)

        out = self.relu2(self.bn2(out0))
        if self.save_features:
            self.feats.append(out.clone().detach())
            self.densities.append((out.data != 0.0).sum().item()/out.numel())
        if self.droprate > 0:
            out = self.dropout(out)
        if self.bench:
            out = self.bench.forward(self.conv2, out, str(self.in_planes) + '.conv2')
        else:
            out = self.conv2(out)

        return self.res(out, x if self.equalInOut else self.convShortcut(x))
        # return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    """Wide Residual Network network block which holds basic blocks.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, save_features=False, bench=None):
        super(NetworkBlock, self).__init__()
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, save_features=self.save_features, bench=self.bench))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
            if self.save_features:
                self.feats += layer.feats
                self.densities += layer.densities
                del layer.feats[:]
                del layer.densities[:]
        return x

############################################################################################################
################################################ ResNet ####################################################
############################################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.res = Residual()
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.res(out, self.shortcut(x))
        # out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.res = Residual()
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        out = self.res(out, self.shortcut(x))
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes, bias=False)
        self.relu = nn.ReLU()
        # self.avg_pool = nn.AvgPool2d(8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.classifier(out)
        # out = self.log_softmax(out)
        return out

class Block_(nn.Module):
    """A ResNet block."""

    def __init__(self, f_in: int, f_out: int, downsample=False):
        super(Block_, self).__init__()

        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f_out)
        self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f_out)
        self.relu = nn.ReLU()
        self.res = Residual()
        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out)
            )
        else:
            # self.shortcut = layers.Identity2d(f_in)
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.res(out, self.shortcut(x))
        return self.relu(out)
    
class ResNet_(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""
    
    def __init__(self, block, plan, num_classes):
        super(ResNet_, self).__init__()

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)
        self.relu = nn.ReLU()
        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):  # plan = [(16,20), (32,20), (64,20)]
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        self.fc = nn.Linear(plan[-1][0], num_classes)
        # self.avg_pool2d = nn.AvgPool2d(8)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1,1))
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()
        self._initialize_weights()


    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        # out = self.log_softmax(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _plan(D, W):
    """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

    The ResNet is structured as an initial convolutional layer followed by three "segments"
    and a linear output layer. Each segment consists of D blocks. Each block is two
    convolutional layers surrounded by a residual connection. Each layer in the first segment
    has W filters, each layer in the second segment has 32W filters, and each layer in the
    third segment has 64W filters.

    The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
    N is the total number of layers in the network: 2 + 6D.
    The default value of W is 16 if it isn't provided.

    For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
    linear layer, there are 18 convolutional layers in the blocks. That means there are nine
    blocks, meaning there are three blocks per segment. Hence, D = 3.
    The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
    """
    if (D - 2) % 3 != 0:
        raise ValueError('Invalid ResNet depth: {}'.format(D))
    D = (D - 2) // 6
    plan = [(W, D), (2*W, D), (4*W, D)]

    return plan

def ResNet20(num_classes=1000):
    plan = _plan(20, 16)
    return ResNet_(Block_, plan, num_classes)

def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2,2,2,2],num_classes)

def ResNet34(num_classes=1000):
    return ResNet(BasicBlock, [3,4,6,3],num_classes)

def ResNet50(num_classes=1000):
    return ResNet(Bottleneck, [3,4,6,3],num_classes)

def ResNet101(num_classes=1000):
    return ResNet(Bottleneck, [3,4,23,3],num_classes)

def ResNet152(num_classes=1000):
    return ResNet(Bottleneck, [3,8,36,3],num_classes)


from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# from efficientnet import * 
from efficientnet import _efficientnet, _efficientnet_conf, EfficientNet, EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights
from efficientnet import EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights


def efficientnet_b0(
    *, weights: Optional[EfficientNet_B0_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """EfficientNet B0 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B0_Weights
        :members:
    """
    weights = EfficientNet_B0_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return _efficientnet(
        inverted_residual_setting, kwargs.pop("dropout", 0.2), last_channel, weights, progress, **kwargs
    )


def efficientnet_b1(
    *, weights: Optional[EfficientNet_B1_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B1_Weights
        :members:
    """
    weights = EfficientNet_B1_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
    return _efficientnet(
        inverted_residual_setting, kwargs.pop("dropout", 0.2), last_channel, weights, progress, **kwargs
    )


def efficientnet_b2(
    *, weights: Optional[EfficientNet_B2_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B2_Weights
        :members:
    """
    weights = EfficientNet_B2_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
    return _efficientnet(
        inverted_residual_setting, kwargs.pop("dropout", 0.3), last_channel, weights, progress, **kwargs
    )


def efficientnet_b3(
    *, weights: Optional[EfficientNet_B3_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B3_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B3_Weights
        :members:
    """
    weights = EfficientNet_B3_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        weights,
        progress,
        **kwargs,
    )


def efficientnet_b4(
    *, weights: Optional[EfficientNet_B4_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B4_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B4_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B4_Weights
        :members:
    """
    weights = EfficientNet_B4_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        weights,
        progress,
        **kwargs,
    )


def efficientnet_b5(
    *, weights: Optional[EfficientNet_B5_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B5_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B5_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B5_Weights
        :members:
    """
    weights = EfficientNet_B5_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b6(
    *, weights: Optional[EfficientNet_B6_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """EfficientNet B6 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B6_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B6_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B6_Weights
        :members:
    """
    weights = EfficientNet_B6_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.5),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b7(
    *, weights: Optional[EfficientNet_B7_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """EfficientNet B7 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B7_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B7_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B7_Weights
        :members:
    """
    weights = EfficientNet_B7_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.5),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_v2_s(
    *, weights: Optional[EfficientNet_V2_S_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_S_Weights
        :members:
    """
    weights = EfficientNet_V2_S_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


def efficientnet_v2_m(
    *, weights: Optional[EfficientNet_V2_M_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-M architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_M_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_M_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_M_Weights
        :members:
    """
    weights = EfficientNet_V2_M_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


def efficientnet_v2_l(
    *, weights: Optional[EfficientNet_V2_L_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_L_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_L_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_L_Weights
        :members:
    """
    weights = EfficientNet_V2_L_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)