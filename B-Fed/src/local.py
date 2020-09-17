import copy

import torch
import torch.optim as optim


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(len(w)))
    return w_avg


def test(model, testloader, criterion, device):
    loss, correct, total = 0, 0, 0

    model.eval()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        batch_ls = criterion(outputs, labels)
        loss += batch_ls.item()

        _, preds = torch.max(outputs, 1)
        preds = preds.view(-1)
        correct += torch.sum(torch.eq(preds, labels)).item()
        total += len(labels)

    accuracy = correct / total
    loss /= len(testloader)
    return accuracy, loss


def train(model, trainloader, cfg, criterion, device):
    losses = []

    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['decay'])

    model.train()

    for ep in range(cfg['epochs']):
        batch_ls = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_ls.append(loss.item())

        loss_avg = sum(batch_ls) / len(batch_ls)
        losses.append(loss_avg)
    return model.state_dict(), losses
