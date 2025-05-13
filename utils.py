import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import glob
from shutil import move
from os.path import join
from os import listdir, rmdir

from torchvision import datasets

# Based on https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py
def TINYIMAGENET(root, train=True, transform=None, target_transform=None, download=False):
    
    def _exists(root, filename):
        return os.path.exists(os.path.join(root, filename))

    def _download(url, root, filename):
        datasets.utils.download_and_extract_archive(url=url, 
                                                    download_root=root, 
                                                    extract_root=root, 
                                                    filename=filename)

    def _setup(root, base_folder):
        target_folder = os.path.join(root, base_folder, 'val/')

        val_dict = {}
        with open(target_folder + 'val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]
                
        paths = glob.glob(target_folder + 'images/*')
        paths[0].split('/')[-1]
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(target_folder + str(folder)):
                os.mkdir(target_folder + str(folder))
                
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            dest = target_folder + str(folder) + '/' + str(file)
            move(path, dest)
            
        os.remove(target_folder + 'val_annotations.txt')
        rmdir(target_folder + 'images')

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'

    if download and not _exists(root, filename):
        _download(url, root, filename)
        _setup(root, base_folder)
    folder = os.path.join(root, base_folder, 'train' if train else 'val')

    return datasets.ImageFolder(folder, transform=transform, target_transform=target_transform)

 
class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""
    def __init__(self,parent_dataset,split_start=-1,split_end= -1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and     split_start < split_end , "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start


    def __getitem__(self,index):
        assert index < len(self),"index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


def get_cifar100_dataloaders(args, validation_split=0.0, num_workers=10):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Data
    print('==> Preparing data..')
    
    
    if 'swap' in args.ablation:
        size = 224
        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True,
                                             transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, test_loader

def get_cifar10_dataloaders(args, validation_split=0.0, num_workers=10):
    """Creates augmented train, validation, and test data loaders."""

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))


    if 'swap' in args.ablation:
        size = 224
        train_transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
            #                                             (4,4,4,4),mode='reflect').squeeze()),
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32),
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        test_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
            #                                             (4,4,4,4),mode='reflect').squeeze()),
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    full_dataset = datasets.CIFAR10(args.data_dir, True, train_transform, download=True)
    test_dataset = datasets.CIFAR10(args.data_dir, False, test_transform, download=False)


    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=num_workers,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=num_workers,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_tinyimagenet_dataloaders(args, validation_split=0.0, num_workers=10):
    mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
    normalize = transforms.Normalize(mean, std)

    if 'swap' in args.ablation:
        size = 224
        train_dataset = TINYIMAGENET(args.data_dir, 
            train=True, download=True, 
            transform=transforms.Compose([
                transforms.Resize(size),
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        val_dataset = TINYIMAGENET(args.data_dir, 
            train=False, download=True, 
            transform=transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = TINYIMAGENET(args.data_dir, 
            train=True, download=True, 
            transform=transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        val_dataset = TINYIMAGENET(args.data_dir, 
            train=False, download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=num_workers)
    return train_loader, val_loader, val_loader

def get_imagenet_dataloaders(args, validation_split=0.0, num_workers=10):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_dataset = datasets.ImageFolder("/cm/archive/thoth/dataset/ImageNet/train", 
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2,1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers)
    

    val_dataset = datasets.ImageFolder("/cm/archive/thoth/dataset/ImageNet/val", 
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=num_workers)
    return train_loader, val_loader, val_loader

def get_fake_dataloaders(args, validation_split=0.0, num_workers=10):
    mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
    normalize = transforms.Normalize(mean, std)

    train_dataset = torchvision.datasets.FakeData(
        size=1000, image_size=(3, 224, 224), num_classes=1000,

        transform=transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FakeData(
        size=1000, image_size=(3, 224, 224), num_classes=1000,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=num_workers)
    return train_loader, val_loader, val_loader

# def get_tinyimagenet_dataloaders(args, validation_split=0.0):
#     traindir = os.path.join(args.datadir, 'train')
#     valdir = os.path.join(args.datadir, 'val')
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))

#     if args.distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#     else:
#         train_sampler = None

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#         num_workers=args.workers, pin_memory=True, sampler=train_sampler)

#     val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(valdir, transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ])),
#         batch_size=args.batch_size, shuffle=False,
#         num_workers=args.workers, pin_memory=True)
#     return train_loader, val_loader

def get_mnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transform=transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.MNIST(f'{args.data_dir}', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(f'{args.data_dir}', train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader


def plot_class_feature_histograms(args, model, device, test_loader, optimizer):
    if not os.path.exists('./results'): os.mkdir('./results')
    model.eval()
    agg = {}
    num_classes = 10
    feat_id = 0
    sparse = not args.dense
    model_name = 'alexnet'
    #model_name = 'vgg'
    #model_name = 'wrn'


    densities = None
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % 100 == 0: print(batch_idx,'/', len(test_loader))
        with torch.no_grad():
            #if batch_idx == 10: break
            data, target = data.to(device), target.to(device)
            for cls in range(num_classes):
                #print('=='*50)
                #print('CLASS {0}'.format(cls))
                model.t = target
                sub_data = data[target == cls]

                output = model(sub_data)

                feats = model.feats
                if densities is None:
                    densities = []
                    densities += model.densities

                if len(agg) == 0:
                    for feat_id, feat in enumerate(feats):
                        agg[feat_id] = []
                        #print(feat.shape)
                        for i in range(feat.shape[1]):
                            agg[feat_id].append(np.zeros((num_classes,)))

                for feat_id, feat in enumerate(feats):
                    map_contributions = torch.abs(feat).sum([0, 2, 3])
                    for map_id in range(map_contributions.shape[0]):
                        #print(feat_id, map_id, cls)
                        #print(len(agg), len(agg[feat_id]), len(agg[feat_id][map_id]), len(feats))
                        agg[feat_id][map_id][cls] += map_contributions[map_id].item()

                del model.feats[:]
                del model.densities[:]
                model.feats = []
                model.densities = []

    if sparse:
        np.save('./results/{0}_sparse_density_data'.format(model_name), densities)

    for feat_id, map_data in agg.items():
        data = np.array(map_data)
        #print(feat_id, data)
        full_contribution = data.sum()
        #print(full_contribution, data)
        contribution_per_channel = ((1.0/full_contribution)*data.sum(1))
        #print('pre', data.shape[0])
        channels = data.shape[0]
        #data = data[contribution_per_channel > 0.001]

        channel_density = np.cumsum(np.sort(contribution_per_channel))
        print(channel_density)
        idx = np.argsort(contribution_per_channel)

        threshold_idx = np.searchsorted(channel_density, 0.05)
        print(data.shape, 'pre')
        data = data[idx[threshold_idx:]]
        print(data.shape, 'post')

        #perc = np.percentile(contribution_per_channel[contribution_per_channel > 0.0], 10)
        #print(contribution_per_channel, perc, feat_id)
        #data = data[contribution_per_channel > perc]
        #print(contribution_per_channel[contribution_per_channel < perc].sum())
        #print('post', data.shape[0])
        normed_data = np.max(data/np.sum(data,1).reshape(-1, 1), 1)
        #normed_data = (data/np.sum(data,1).reshape(-1, 1) > 0.2).sum(1)
        #counts, bins = np.histogram(normed_data, bins=4, range=(0, 4))
        np.save('./results/{2}_{1}_feat_data_layer_{0}'.format(feat_id, 'sparse' if sparse else 'dense', model_name), normed_data)
        #plt.ylim(0, channels/2.0)
        ##plt.hist(normed_data, bins=range(0, 5))
        #plt.hist(normed_data, bins=[(i+20)/float(200) for i in range(180)])
        #plt.xlim(0.1, 0.5)
        #if sparse:
        #    plt.title("Sparse: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_sp.png'.format(feat_id))
        #else:
        #    plt.title("Dense: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_d.png'.format(feat_id))
        #plt.clf()
