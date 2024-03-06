import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
import logging

criterion_kl = nn.KLDivLoss(size_average=False)
mart_kl = nn.KLDivLoss(reduction='none')


def BPTT_attack(model, image, T, surrogate='PCW', gamma=1.0):
    model.set_simulation_time(T, mode='bptt')
    model.set_surrogate_gradient(surrogate=surrogate, gamma=gamma, mode='bptt')
    output = model(image).mean(0)
    return output


def BPTR_attack(model, image, T, surrogate='PCW', gamma=1.0):
    model.set_simulation_time(T, mode='bptr')
    output = model(image).mean(0)
    model.set_simulation_time(T)
    return output


def Act_attack(model, image, T, surrogate='PCW', gamma=1.0):
    model.set_simulation_time(0)
    output = model(image)
    model.set_simulation_time(T)
    return output


def val(model, test_loader, device, T, adv_train=None):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        if adv_train is not None:
            adv_train.set_model_training_mode(model_training=False,
                                              batchnorm_training=False,
                                              dropout_training=False)
            inputs = adv_train(inputs, targets.to(device))
            model.set_simulation_time(T)

        with torch.no_grad():
            outputs = model(inputs).mean(0) if T > 0 else model(inputs)

        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    return final_acc


def train(model, device, train_loader, criterion, optimizer, T, adv_train, trades_beta=0., mart_beta=0.):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, data in enumerate(train_loader):
        images, labels = data[0].to(device), data[1].to(device)
        batch_size = images.shape[0]
        optimizer.zero_grad()

        if trades_beta != 0. or mart_beta != 0.:
            outputs_clean = model(images).mean(0) if T > 0 else model(images)
            loss_natural = criterion(outputs_clean, labels)

        if adv_train is not None:
            adv_train.set_model_training_mode(model_training=False,
                                              batchnorm_training=False,
                                              dropout_training=False)
            images_adv = adv_train(images, labels)
            outputs = model(images_adv).mean(0) if T > 0 else model(images_adv)
        else:
            outputs = model(images).mean(0) if T > 0 else model(images)

        if trades_beta != 0.:
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(outputs_clean, dim=1))
            loss = loss_natural + trades_beta * loss_robust
        else:
            if mart_beta != 0.:
                adv_probs = F.softmax(outputs, dim=1)
                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
                loss_adv = F.cross_entropy(outputs, labels) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
                nat_probs = F.softmax(outputs_clean, dim=1)
                true_probs = torch.gather(nat_probs, 1, (labels.unsqueeze(1)).long()).squeeze()
                loss_robust = (1.0 / batch_size) * torch.sum(
                    torch.sum(mart_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
                loss = loss_adv + float(mart_beta) * loss_robust
            else:
                loss = criterion(outputs, labels)

        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()

        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
