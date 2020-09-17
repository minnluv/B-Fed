# Boosted Federated Learning by Forgetting Event
## Overview
B-Fed algorithm is designed for federated learning of modern convolutional neural network architectures. B-Fed tested the influence of forgettable examples in federated learning. Because forgetting event happens more often in federated learning, B-Fed increases the chance of learning from randomly chosen forgettable examples in each batch.

+ Before: Insert a certain number of forgettable examples in batches
+ Update: Each batch is randomly constructed, but a certain number of forgettable examples is given another chance to be learned

## Dependencies
Tested stable dependencies:

+ Python 3.7.9
+ PyTorch 1.6.0
+ Torchvision 0.7.0
+ Scikit-Learn 0.23.2
+ NumPy 1.19.2
+ EfficientNet-PyTorch 0.7.0 (https://github.com/lukemelas/EfficientNet-PyTorch.git)
+ CUDA 10.2

## Execution
### Arguments
| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| seed | int | Seed for random number generator | 42 |
| num | int | Number of virtual parties to use in federated environment | 16 |
| model | str | CNN model to use [lenet] / [vgg] / [resnet] / [mobilenet] / [densenet] / [efficientnet] | vgg |
| dataset | str | Dataset to use [mnist] / [cifar10] | cifar10 |
| boost | float | Fraction of forgettable examples to boost | 0.3 |
| ifd | bool | Identical forgettable example distribution among parties | True |
| stratify | bool | Stratify local data by label | True |
| rounds | int | Number of communication rounds | 50 |
| epochs | int | Number of local epochs | 10 |
| batch | int | Batch size | 64 |
| lr | float | Learning rate | 0.01 |
| momentum | float | Momentum for SGD | 0.9 |
| decay | float | Weight decay for SGD | 0.0001 |
| gpu | int | GPU to use | 0 |
| store | bool | Store configurations and results | True |
| filename | str | File name used to save (No extension) | None |
| config | str | Overwrite configurations with JSON config file instead of arguments (No extension) | None |

* Note: Having `--dataset=mnist` will result in running with LeNet-5 model regradless of model argument.

### Sample command
```
python main.py --seed=42 \
--num=16 \
--model=vgg \
--dataset=cifar10 \
--boost=0.3 \
--ifd=True \
--stratify=True \
--rounds=50 \
--epochs=10 \
--batch=64 \
--lr=0.01 \
--momentum=0.9 \
--decay=0.0001 \
--gpu=0 \
--store=True \
--filename=TmpRun
```
### Sample command using JSON config file
```
python main.py --gpu=0 \
--store=True \
--filename=TryOut \
--config=case4
```