import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

PATH_SRC = os.getcwd()
PATH_ROOT = os.path.dirname(PATH_SRC)
PATH_DATA = os.path.join(PATH_ROOT, 'data')
if not os.path.exists(PATH_DATA):
    os.mkdir(PATH_DATA)

def get_dataset(cfg):
    if cfg['dataset'] == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10(PATH_DATA, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(PATH_DATA, train=False, transform=transform, download=False)

    elif cfg['dataset'] == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(PATH_DATA, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(PATH_DATA, train=False, transform=transform, download=False)

    return train_dataset, test_dataset


def split_dataset(cfg):
    df = pd.read_csv(os.path.join(PATH_DATA, f'{cfg["dataset"]}_flagged.csv'), encoding='utf-8')
    forget_df = df.loc[df['forgettable'] == 1]
    unforget_df = df.loc[df['forgettable'] == 0]

    parties = []

    if cfg['ifd']:
        forget_items = int(len(forget_df.index) / cfg['num'])
        unforget_items = int(len(unforget_df.index) / cfg['num'])
        num_items = forget_items + unforget_items

        tmpf = forget_df.copy()
        tmpu = unforget_df.copy()

        for i in range(cfg['num'] - 1):
            if cfg['stratify']:
                f, _ = train_test_split(tmpf['indices'], train_size=forget_items, random_state=cfg['seed'],
                                        shuffle=True,
                                        stratify=tmpf['targets'])
                u, _ = train_test_split(tmpu['indices'], train_size=unforget_items, random_state=cfg['seed'],
                                        shuffle=True,
                                        stratify=tmpu['targets'])
            else:
                f, _ = train_test_split(tmpf['indices'], train_size=forget_items, random_state=cfg['seed'],
                                        shuffle=True)
                u, _ = train_test_split(tmpu['indices'], train_size=unforget_items, random_state=cfg['seed'],
                                        shuffle=True)
            tmpf = tmpf.drop(f)
            tmpu = tmpu.drop(u)
            p = pd.concat([f, u])
            p = p.to_numpy()
            parties.append(df.iloc[p])

        f = tmpf['indices']
        u = tmpu['indices']
        p = pd.concat([f, u])
        p = p.to_numpy()
        parties.append(df.iloc[p])

    else:
        tmp = df.copy()
        num_items = int(len(tmp.index) / cfg['num'])

        for i in range(cfg['num'] - 1):
            if cfg['stratify']:
                p, _ = train_test_split(tmp['indices'], train_size=num_items, random_state=cfg['seed'], shuffle=True,
                                        stratify=tmp['targets'])
            else:
                p, _ = train_test_split(tmp['indices'], train_size=num_items, random_state=cfg['seed'], shuffle=True)
            tmp = tmp.drop(p)
            p = p.to_numpy()
            parties.append(df.iloc[p])

        p = tmp['indices']
        parties.append(df.iloc[p])

    print(f'\n| {cfg["dataset"].upper()} |')
    print('|-- Forgettable samples: ', len(forget_df.index))
    print('|-- Unforgettable samples: ', len(unforget_df.index))

    print('| Number of parties: ', cfg['num'])
    print('|-- Number of local data: ', num_items)
    print('|-- Identical forgettable distribution: ', cfg['ifd'])
    print('|-- Stratify: ', cfg['stratify'])

    return parties


def boost_examples(parties, cfg):
    boosted = []
    for p in parties:
        p.reset_index(drop=True, inplace=True)
        boost = p.iloc[p[p['forgettable'] == 1].sample(frac=cfg['boost']).index]
        p = p.append(boost, ignore_index=True)
        boosted.append(p)
    return boosted


def get_loaders(train_dataset, test_dataset, parties, cfg):
    trainloaders = []
    for p in parties:
        train_subset = Subset(train_dataset, p['indices'])
        trainloaders.append(DataLoader(train_subset, batch_size=cfg['batch'], shuffle=True))
    testloader = DataLoader(test_dataset, batch_size=cfg['batch'], shuffle=False, num_workers=0)
    return trainloaders, testloader
