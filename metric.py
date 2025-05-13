import torch
import torch.nn as nn
import numpy as np

def flop(model, ones, device):

    total = [0]
    def count_flops(name):
        def hook(module, input, output):
            flops = {}
            if isinstance(module, nn.Linear) or isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                # total[0] += in_features * out_features * module.sparsity
                total[0] += module.mask.sum().item()
                if module.bias is not None:
                    total[0] += out_features
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = int(np.prod(module.kernel_size))
                output_size = output.size(2) * output.size(3)
                # total[0] += in_channels * out_channels * kernel_size * output_size * module.sparsity
                total[0] += output_size * module.mask.sum().item()
                if module.bias is not None:
                    total[0] += out_channels * output_size
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    total[0] += module.num_features
                    total[0] += module.num_features
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm2d):
                output_size = output.size(2) * output.size(3)
                if module.affine:
                    total[0] += module.num_features * output_size
                    total[0] += module.num_features * output_size
            
        return hook
    
    for name, module in model.named_modules():
        module.register_forward_hook(count_flops(name))

    model(ones)

    return total[0]