import json

import numpy as np


def store_config(cfg, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, separators=(',', ': '))


def store_results(train_losses, test_losses, test_accs, path):
    with open(f'{path}_tr_ls.npy', 'wb') as f:
        np.save(f, train_losses)
    with open(f'{path}_te_ls.npy', 'wb') as f:
        np.save(f, test_losses)
    with open(f'{path}_te_acc.npy', 'wb') as f:
        np.save(f, test_accs)
