import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from  models import Residual
from efficientnet import StochasticDepth, SqueezeExcitation
import wandb
import math


class TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:k]] = 0.0
        flat_out[idx[k:]] = 1.0

        return out
    
    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    

def NPB_objective(ones, model, args, score_optimizer, lr_score_scheduler, update=True, adjust_lr=False):
    model.zero_grad()

    if 'layer-wise' in args.ablation:
        score = torch.cat([m.score.flatten() for m in model.NPB_modules])
        mask = TopK.apply(score.abs(), int((model.sparsity) * score.numel()))
        
        post = 0
        for m in model.NPB_modules:
            m.mask = mask[post: post+m.score.numel()].view_as(m.score)
            post += m.score.numel()
            print(m.mask.sum().item(), end=" ")
        print()

    eff_params = 0
    eff_nodes = 0
    eff_kernels = 0


    model.apply(lambda m: setattr(m, "npb", True))
    model.apply(lambda m: setattr(m, "learn_mask", True))
    
    model.zero_grad()

    eff_paths = model(ones)
    cum_max_paths = 0
    for m in model.NPB_modules:
        if hasattr(m, "max_paths"):
            cum_max_paths += m.max_paths.log()
            m.max_paths = 0 
    eff_paths = eff_paths.sum().log() + cum_max_paths

    eff_paths.backward()

    all_layer_eff_nodes = []

    path_grads = []
    node_grads = []
    kernel_grads = []

    for i, m in enumerate(model.NPB_modules): 

        path_grad = torch.zeros_like(m.score) if m.score.grad is None else m.score.grad

        layer_eff_params = path_grad.abs() * m.mask
        layer_eff_nodes = layer_eff_params.sum(m.dim_out).view(m.view_out)
        layer_eff_nodes_hard = (layer_eff_nodes > 0).float()
        all_layer_eff_nodes += [layer_eff_nodes_hard.sum().item()]
        eff_nodes += layer_eff_nodes_hard.sum()

        if len(m.weight.shape) == 4:
            layer_eff_kernels = layer_eff_params.sum((2,3)).unsqueeze(2).unsqueeze(3)
            layer_eff_kernels_hard = (layer_eff_kernels > 0).float()
            eff_kernels += layer_eff_kernels_hard.sum()
        else:
            layer_eff_params_hard = (layer_eff_params > 0).float()
            eff_kernels += layer_eff_params_hard.sum()

        layer_eff_params_hard = (layer_eff_params > 0).float()
        eff_params += layer_eff_params_hard.sum()

        node_grad = path_grad * (1 - layer_eff_nodes_hard)
        kernel_grad = path_grad * (1 - layer_eff_kernels_hard) if len(m.weight.shape) == 4 else path_grad * (1 - layer_eff_params_hard)

        path_grads.append(path_grad)
        node_grads.append(node_grad)
        kernel_grads.append(kernel_grad)

        if update:
            m.score.grad = - ((1-args.alpha) * path_grad + args.alpha * ((1-args.beta) * node_grad + args.beta * kernel_grad))
 
    if update:       
        score_optimizer.step()

    return eff_paths, eff_nodes, eff_kernels, eff_params, all_layer_eff_nodes

# Get sub network

def get_mask_by_weight(self):
    return TopK.apply(self.weight.abs(), self.num_zeros)

def get_mask_by_score(self):
    return TopK.apply(self.score.abs(), self.num_zeros)

def get_weight(self):
    return self.weight

def get_masked_weight(self):
    if self.learn_mask:
        # self.mask = self.get_mask()
        self.register_buffer('mask', self.get_mask())
        return self.mask * self.weight
    else:
        return self.mask * self.weight

def linear_forward(self, x, weight, bias):
    return F.linear(x, weight, bias)

# Effectively compute the number of effective paths via forward pass

def NPB_forward(self, x):
    if self.npb:
        if self.learn_mask:
            self.register_buffer('mask', self.get_mask())

        self.max_paths = x.max()
        x = self.base_func(x / self.max_paths, self.mask, None)
        return x
    else:
        return self.base_func(x, self.get_weight(), self.bias)
    
def NPB_dummy_forward(self, x):
    if self.npb:
        return x
    else:
        return self.original_forward(x)
    

def NPB_register(model, args):
    model.apply(lambda m: setattr(m, "npb", False))
    model.apply(lambda m: setattr(m, "learn_mask", False))
    NPB_modules = []

    for m in model.modules():
        if hasattr(m, "pos_embedding"):
            m.pos_embedding = nn.Parameter(torch.zeros_like(m.pos_embedding))

        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            NPB_modules.append(m)
            setattr(m, 'original_forward', m.forward)

            m.score = nn.Parameter(torch.empty_like(m.weight), requires_grad=True).to(args.device)
            nn.init.normal_(m.score, 0, 1)
            setattr(m, 'get_weight', get_masked_weight.__get__(m, m.__class__))
            setattr(m, 'get_mask', get_mask_by_score.__get__(m, m.__class__))

            m.sparsity = args.sparsity
            m.num_zeros = int((m.sparsity) * m.score.numel())
            m.register_buffer('mask', m.get_mask())
            
            if isinstance(m, nn.Linear):
                m.dim_in = (0)
                m.dim_out = (1)
                m.view_in = (1, -1)
                m.view_out = (-1, 1)
                setattr(m, 'base_func', linear_forward.__get__(m, m.__class__))
            else:
                m.dim_in = (0,2,3)
                m.dim_out = (1,2,3)
                m.view_in = (1, -1, 1, 1)
                m.view_out = (-1, 1, 1, 1)
                setattr(m, 'base_func', m._conv_forward)

            setattr(m, 'forward', NPB_forward.__get__(m, m.__class__))

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.LogSoftmax) or isinstance(m, nn.ReLU) \
            or isinstance(m, nn.Dropout) or isinstance(m, nn.SiLU) or isinstance(m, StochasticDepth) or isinstance(m, nn.Sigmoid) \
            or isinstance(m, SqueezeExcitation):
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_dummy_forward.__get__(m, m.__class__))

    model.NPB_modules = NPB_modules
    model.weights = [m.weight for m in model.NPB_modules]
    model.scores = [m.score for m in model.NPB_modules]
    if 'erk' not in args.ablation:
        ERK_sparsify(model, args.sparsity)
    model.sparsity = args.sparsity

def post_pruning(model):
    for m in model.NPB_modules:
        m.mask = m.mask.detach().clone()
        m.cache_in = None
        m.cache_out = None
    model.zero_grad()

def fired_mask_update(model):
    for m in model.NPB_modules:
        m.fired_mask = m.mask.data.byte() | m.fired_mask.data.byte()

def fired_weights_summary(model):
    ntotal_fired_weights = 0.0
    ntotal_weights = 0.0
    for m in model.NPB_modules:
        ntotal_fired_weights += float(m.fired_mask.sum().item())
        ntotal_weights += float(m.fired_mask.numel())

    total_fired_weights = ntotal_fired_weights/ntotal_weights
    print('The percentage of the total fired weights is:', total_fired_weights)
    return total_fired_weights

def ERK_sparsify(model, sparsity=0.9):
    print('initialize by ERK')
    density = 1 - sparsity
    erk_power_scale = 1

    total_params = 0
    for m in model.NPB_modules:
        total_params += m.score.numel()
    is_epsilon_valid = False

    dense_layers = set()
    while not is_epsilon_valid:
        divisor = 0
        rhs = 0
        for m in model.NPB_modules:
            m.raw_probability = 0
            n_param = np.prod(m.score.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density

            if m in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                m.raw_probability = (np.sum(m.score.shape) / np.prod(m.score.shape)) ** erk_power_scale
                divisor += m.raw_probability * n_param

        epsilon = rhs / divisor
        max_prob = np.max([m.raw_probability for m in model.NPB_modules])
        # print([m.raw_probability for m in model.NPB_modules])
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for m in model.NPB_modules:
                if m.raw_probability == max_prob:
                    dense_layers.add(m)
        else:
            is_epsilon_valid = True

    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for i, m in enumerate(model.NPB_modules):
        n_param = np.prod(m.score.shape)
        if m in dense_layers:
            m.sparsity = 0
        else:
            probability_one = epsilon * m.raw_probability
            m.sparsity = 1 - probability_one
        m.num_zeros = int((m.sparsity) * m.score.numel())
        # m.num_zeros = max(m.num_zeros, 1)
        # print(
        #     f"layer: {i}, shape: {m.score.shape}, sparsity: {m.sparsity}"
        # )
        total_nonzero += (1-m.sparsity) * m.score.numel()
    print(f"Overall sparsity {1-total_nonzero / total_params}")
