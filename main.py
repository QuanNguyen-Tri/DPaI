import argparse
import os
import time
import torch
import torchvision
from torchvision.datasets import CIFAR100, CIFAR10
from npb import *
import models
import wandb
from utils import *
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
from metric import *
from scheduler import WarmupLinearSchedule, WarmupCosineSchedule

def train(args, global_step, model, device, train_loader, optimizer, lr_scheduler, epoch, masks=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    enabled = False

    if args.amp:
        enabled = True
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if 'dst' not in  args.method:
            with torch.cuda.amp.autocast(enabled=enabled):
                output = model(data)
                loss = F.cross_entropy(output, target)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step() 

            if "ViT" in args.model:
                lr_scheduler.step() 

            i = 0
            for m in model.module.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    m.weight.data = m.weight.data * masks[i].to(m.weight.data.device)
                    i += 1

            if (global_step + 1) % 100 == 0:
                print(f"Step {global_step} / {args.total_steps}, lr {optimizer.param_groups[0]['lr']}, loss {train_loss/batch_idx}")

            global_step += 1
            if global_step >= args.total_steps:
                break


        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]
        

    # training summary
    train_acc = 100. * correct / float(n)
    train_loss = train_loss/batch_idx

    print('\n Training summary: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%) \n'.format(
        epoch , train_loss, correct, n, train_acc))

    if args.wandb:
        wandb.log({'train acc': train_acc, 'train loss': train_loss, 'epoch':epoch})

    return global_step

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    model.apply(lambda m: setattr(m, "npb", False))
    model.apply(lambda m: setattr(m, "learn_mask", False))
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)

def main():
    parser = argparse.ArgumentParser(description='Pruning at Initialization')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_score', type=float, default=0.1, metavar='LR',
                        help='learning rate score (default: 0.1)', required=False)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--lr_drop_rate', type=float, default=0.1)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--amp', action='store_true', help='torch mix precision.')
    parser.add_argument('--valid_split', type=float, default=0.0)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--heuristic', type=str, default='none')
    parser.add_argument('--ablation', type=str, default='none')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--wandb_dir', type=str, default='wandb')
    parser.add_argument('--wandb_project', type=str, default='DPaI')
    parser.add_argument('--wandb_key', type=str, default='', help='login wandb')
    parser.add_argument('--wandb', action='store_true', help='logging results with wandb')
    parser.add_argument('--save', action='store_true', help='save model')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--alpha', type=float, default=0, required=False, help='node path balancing hyperparameter')
    parser.add_argument('--beta', type=float, default=0, required=False)
    parser.add_argument('--gamma', type=float, default=0, required=False)
    parser.add_argument('--eta', type=float, default=0, required=False)
    parser.add_argument('--lamb', type=float, default=0, required=False)
    parser.add_argument('--warmup', type=float, default=10000, required=False)
    parser.add_argument('--total_steps', type=int, default=100000, required=False)
    parser.add_argument('--sparsity', type=float, default=0.99)
    parser.add_argument('--compression', type=float, default=-1)
    parser.add_argument('--num_steps', type=int, default=500, help='number of npb optimization steps')
    parser.add_argument('--method', type=str, default='pai')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--lr-drops', type=int, nargs='*', default=[60, 120],
                        help='list of learning rate drops (default: [])')
    

    args = parser.parse_args()
    name = f'{args.model}_{args.data}_{args.method}_{args.compression}_alpha_{args.alpha}_beta_{args.beta}_lamb_{args.lamb}_seed_{args.seed}_lr_score_{args.lr_score}_{args.heuristic}_{args.ablation}'

    if args.compression != -1:
        args.sparsity = 1-10**(-float(args.compression))

    print('Sparisty:', args.sparsity, "learning rate score", args.lr_score)

    # if 'cuda' in args.device:
    #     torch.cuda.set_device(args.gpu_id)
    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    args.reproducibility = True
    if args.reproducibility: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    

    if args.wandb:
        wandb.login(key=args.wandb_key)
        run = wandb.init(
            project=args.wandb_project,
            name=name,
            config=args,
            dir=args.wandb_dir
        )

    if args.data == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, args.valid_split)
        c = 10
        ones = torch.ones((1, 1, 28, 28)).float().to(args.device)
    elif args.data == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split)
        c = 10
        ones = torch.ones((1, 3, 32, 32)).float().to(args.device)
    elif args.data == 'cifar100':
        train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split)
        c = 100
        ones = torch.ones((1, 3, 32, 32)).float().to(args.device)
    elif args.data == 'tiny-imagenet':
        train_loader, valid_loader, test_loader = get_tinyimagenet_dataloaders(args, args.valid_split, num_workers=24)
        c = 200
        ones = torch.ones((1, 3, 64, 64)).float().to(args.device)
    elif args.data == 'imagenet':
        train_loader, valid_loader, test_loader = get_imagenet_dataloaders(args, args.valid_split, num_workers=24)
        c = 1000
        ones = torch.ones((1, 3, 224, 224)).float().to(args.device)
    else:
        train_loader, valid_loader, test_loader = get_fake_dataloaders(args, args.valid_split)
        c = 1000
        ones = torch.ones((1, 3, 224, 224)).float().to(args.device)


    print(f'Pruning {args.model}, data {args.data}')
    score_optimizer = None

    if "ViT" in args.model:
        
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        if "pretrained" in args.model:
            model = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = vit_b_16()

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=False)
        elif args.optimizer == 'adam':
            # optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')
        
        # args.t_total = 100000
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup, t_total=args.total_steps)
        
    else:
        model_class = getattr(models, args.model)
        model = model_class(num_classes=c)
        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=False)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        args.total_steps = 99999999
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate, last_epoch=-1)
    
    model.to(args.device)
    print('Base network number of parameters:', sum([p.numel() for n, p in model.named_parameters() if 'score' not in n]))
    print('Sparse network number of parameters:', int((1-args.sparsity) * sum([p.numel() for n, p in model.named_parameters() if 'score' not in n])))

    start = time.time()
    model.eval()
    if args.method == 'npb':
        ('Pruning at Initialization')
        # args.num_steps = 1000
        NPB_register(model, args)

        scores = [m.score for m in model.NPB_modules]
        score_optimizer = torch.optim.Adam(scores, lr=args.lr_score)
        lr_score_scheduler = torch.optim.lr_scheduler.MultiStepLR(score_optimizer, milestones=[int(args.num_steps / 2), int(args.num_steps * 3 / 4)], gamma=0.5, last_epoch=-1)


        eff_paths, eff_nodes, eff_kernels, eff_params, all_layer_eff_nodes = NPB_objective(ones, model, args, score_optimizer, lr_score_scheduler, update=False, adjust_lr=False)
        print(f'log eff paths: {round(eff_paths.item(), 2)}, eff nodes: {int(eff_nodes)}, eff kernels: {int(eff_kernels)}, eff params: {int(eff_params)}')
        print(f'Layer eff nodes: {all_layer_eff_nodes}')
        loss = ((1-args.alpha) * eff_paths + args.alpha * ((1-args.beta) * eff_nodes.log() + args.beta * eff_kernels.log())).item()
        node_kernel = eff_nodes.log() + eff_kernels.log()
        if args.wandb:
            wandb.log({'eff nodes': eff_nodes, 'eff paths': eff_paths, 'eff kernels': eff_kernels, 'eff params': eff_params, 'loss': loss, 'eff nodes and kernels': node_kernel})
                    
        
        for m in model.NPB_modules:
            m.mask = m.mask.detach().clone()

        last_eff_paths, last_eff_nodes, last_eff_kernels, last_eff_params = eff_paths, eff_nodes, eff_kernels, eff_params
        step = 0
        best_loss = 0

        if 'random' not in args.ablation:
            while True:
                step += 1
                eff_paths, eff_nodes, eff_kernels, eff_params, all_layer_eff_nodes = NPB_objective(ones, model, args, score_optimizer, lr_score_scheduler, update=True, adjust_lr=True)
                loss = ((1-args.alpha) * eff_paths + args.alpha * ((1-args.beta) * eff_nodes.log() + args.beta * eff_kernels.log())).item()
                node_kernel = eff_nodes.log() + eff_kernels.log()
                
                if args.wandb:
                    wandb.log({'eff nodes': eff_nodes, 'eff paths': eff_paths, 'eff kernels': eff_kernels, 'eff params': eff_params, 'loss': loss, 'eff nodes and kernels': node_kernel})
                    
                if step % 10 == 0:
                    print(f'Iter {step} / {args.num_steps}, Loss: {loss}, log eff paths: {round(eff_paths.item(), 2)}, eff nodes: {int(eff_nodes)}, eff kernels: {int(eff_kernels)}, eff params: {int(eff_params)}')
                    print('node, kernel', node_kernel.item())
                   
                if eff_nodes < 1:
                    print(eff_nodes)
                    import sys
                    sys.exit(0)

                if step >= args.num_steps: break

        args.heuristic = None
        eff_paths, eff_nodes, eff_kernels, eff_params, all_layer_eff_nodes = NPB_objective(ones, model, args, score_optimizer, lr_score_scheduler, update=False, adjust_lr=False)
        loss = ((1-args.alpha) * eff_paths + args.alpha * ((1-args.beta) * eff_nodes.log() + args.beta * eff_kernels.log())).item()
        node_kernel = eff_nodes.log() + eff_kernels.log()
        print(f'Best Loss: {loss}, log eff paths: {round(eff_paths.item(), 2)}, eff nodes: {int(eff_nodes)}, eff kernels: {int(eff_kernels)}, eff params: {int(eff_params)}')
        print(f'Layer eff nodes: {all_layer_eff_nodes}')
        if args.wandb:
            wandb.log({'eff nodes': eff_nodes, 'eff paths': eff_paths, 'eff kernels': eff_kernels, 'eff params': eff_params, 'eff nodes and kernels': node_kernel})


        masks = []
        for m in model.NPB_modules:
            masks.append(m.get_mask().detach().clone())
            m.register_buffer('mask', m.get_mask().detach().clone())

        model.apply(lambda m: setattr(m, "npb", False))
        model.apply(lambda m: setattr(m, "learn_mask", False))

        model.eff_paths = eff_paths.item()
        model.eff_nodes = int(eff_nodes)
        model.eff_kernels = int(eff_kernels)
        model.eff_params = int(eff_params)
        model.all_layer_eff_nodes = all_layer_eff_nodes

        if 'reinit' in args.ablation:
            model._initialize_weights()
    
    end = time.time()
    print('Pruning time:', end-start)    

    if "ViT" in args.model:
        
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        if "pretrained" in args.model:
            model = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = vit_b_16()

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=False)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')
        
        # args.t_total = 100000
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup, t_total=args.total_steps)
        
    else:
        model_class = getattr(models, args.model)
        model = model_class(num_classes=c)
        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=False)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        args.total_steps = 99999999
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate, last_epoch=-1)
    

    i = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data * masks[i].to(m.weight.data.device)
            i += 1

    model.masks = masks
    model = torch.nn.DataParallel(model)
    model.to(args.device)


    best_acc = 0
    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        global_step = train(args, global_step, model, args.device, train_loader, optimizer, lr_scheduler, epoch, masks)
        if "ViT" not in args.model:
            lr_scheduler.step()
       
        val_acc = evaluate(args, model, args.device, test_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            
        print('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
        print(f'Best Accuracy {best_acc}.\n')
    
        
        if args.wandb:
            wandb.log({'epoch':epoch, 'val acc': val_acc})

        if global_step >= args.total_steps: break

    print('Testing model')
    print('Best Valid:', best_acc)
    if args.wandb:
        wandb.log({'best valid': best_acc})
    test_acc = evaluate(args, model, args.device, test_loader, is_test_set=True)

    model.apply(lambda m: setattr(m, "npb", False))
    model.apply(lambda m: setattr(m, "learn_mask", False))
    model.test_acc = best_acc

    if args.wandb:
        wandb.log({'eff nodes': eff_nodes, 'eff paths': eff_paths, 'eff kernels': eff_kernels, 'eff params': eff_params, 'eff nodes and kernels': node_kernel})
        wandb.log({'test': test_acc})
        wandb.finish()

if __name__ == '__main__':
   main()
