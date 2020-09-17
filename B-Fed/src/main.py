import copy
import json
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from dataset import get_dataset, split_dataset, boost_examples, get_loaders
from local import average_weights, train, test
from models import MNISTLeNet5, LeNet5, VGG9, ResNet18, MobileNetV2, DenseNet
from options import args
from utils import store_config, store_results

# Path
PATH_SRC = os.getcwd()
PATH_ROOT = os.path.dirname(PATH_SRC)
PATH_CONF = os.path.join(PATH_ROOT, 'configs')
if not os.path.exists(PATH_CONF):
    os.mkdir(PATH_CONF)

# Get configurations
if args.config:
    with open(os.path.join(PATH_CONF, f'{args.config}.json'), 'r') as f:
        cfg = json.load(f)
else:
    cfg = vars(args)

# Seed
os.environ['PYTHONHASHSEED'] = str(cfg['seed'])
random.seed = cfg['seed']
np.random.seed = cfg['seed']
torch.manual_seed = cfg['seed']

# File name
if args.filename:
    fname = args.filename
else:
    fname = 'P{}_{}_BT{}_BS{}_R{}_E{}_{}_{}_S{}'.format(
        cfg['num'], cfg['model'], int(cfg['boost'] * 10), cfg['batch'], cfg['rounds'], cfg['epochs'],
        'FSTR' if cfg['ifd'] else 'FRND', 'LSTR' if cfg['stratify'] else 'LRND', cfg['seed']
    )

print(f'\n>>>> {fname}\n')

# Device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Get data and split by parties
    train_dataset, test_dataset = get_dataset(cfg)
    parties = split_dataset(cfg)
    parties = boost_examples(parties, cfg)
    trainloaders, testloader = get_loaders(train_dataset, test_dataset, parties, cfg)

    # Initialize global model
    if cfg['dataset'] == 'mnist':
        fed_model = MNISTLeNet5()
    elif cfg['dataset'] == 'cifar10':
        if cfg['model'] == 'lenet':
            fed_model = LeNet5()
        elif cfg['model'] == 'vgg':
            fed_model = VGG9()
        elif cfg['model'] == 'resnet':
            fed_model = ResNet18()
        elif cfg['model'] == 'mobilenet':
            fed_model = MobileNetV2()
        elif cfg['model'] == 'densenet':
            fed_model = DenseNet()
        elif cfg['model'] == 'efficientnet':
            fed_model = EfficientNet.from_name('efficientnet-b0')

    fed_model.to(device)
    fed_weights = fed_model.state_dict()
    criterion = nn.CrossEntropyLoss().to(device)
    train_losses, test_accs, test_losses = [], [], []

    # Begin federated learning
    st = time.time()
    for r in range(cfg['rounds']):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {r + 1} / {cfg["rounds"]} |')

        fed_model.train()

        for i, p in enumerate(parties):
            w, ls = train(copy.deepcopy(fed_model), trainloaders[i], cfg, criterion, device)
            local_weights.append(copy.deepcopy(w))
            train_losses.append(ls)
            print('  |-- [Party {:>2}] Average Train Loss: {:.4f} ... {} local epochs'.format(i + 1, sum(ls) / len(ls),
                                                                                              cfg['epochs']))

        fed_weights = average_weights(local_weights)
        fed_model.load_state_dict(fed_weights)

        test_acc, test_ls = test(fed_model, testloader, criterion, device)
        test_accs.append(test_acc)
        test_losses.append(test_ls)
        print('    |---- Test Accuracy: {:.4f}%'.format(100 * test_acc))
        print('    |---- Test Loss: {:.4f}'.format(test_ls))
        print('    |---- Elapsed time: {}'.format(timedelta(seconds=time.time() - st)))

    if args.store:
        PATH_SAVE = os.path.join(PATH_ROOT, 'saves')
        if not os.path.exists(PATH_SAVE):
            os.mkdir(PATH_SAVE)
        store_config(cfg, os.path.join(PATH_CONF, f'{fname}.json'))
        store_results(train_losses, test_losses, test_accs, os.path.join(PATH_SAVE, f'{fname}'))
