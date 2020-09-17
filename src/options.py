import argparse

parser = argparse.ArgumentParser(description='Configurations')

# Environment
parser.add_argument('-s', '--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('-n', '--num', type=int, default=16,
                    help='Number of virtual parties to use in federated environment')
parser.add_argument('-m', '--model', type=str, default='vgg',
                    help='Model [lenet] / [vgg] / [resnet] / [mobilenet] / [densenet]')

# Data
parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                    help='Dataset [mnist] / [cifar10]')
parser.add_argument('-f', '--boost', type=float, default=0.3,
                    help='Fraction of forgettable examples to boost')
parser.add_argument('-i', '--ifd', type=bool, default=True,
                    help='Identical forgettable sample distribution among parties')
parser.add_argument('-t', '--stratify', type=bool, default=True,
                    help='Stratified local data')

# Training
parser.add_argument('-r', '--rounds', type=int, default=50,
                    help='Number of communication rounds')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Number of local epochs in training')
parser.add_argument('-b', '--batch', type=int, default=64,
                    help='Mini-batch size')
parser.add_argument('-l', '--lr', type=float, default=0.01,
                    help='Learning rate')
parser.add_argument('-o', '--momentum', type=float, default=0.9,
                    help='Momentum')
parser.add_argument('-w', '--decay', type=float, default=0.0001,
                    help='Weight decay')

# Others
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU designation')
parser.add_argument('--store', type=bool, default=True,
                    help='Store configs and results')
parser.add_argument('--filename', type=str, default=None,
                    help='File name to save (No extension)')
parser.add_argument('--config', type=str, default=None,
                    help='Get JSON config file instead of arguments (No extension)')

args = parser.parse_args()
