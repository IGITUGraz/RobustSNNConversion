import argparse
import os
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import attack
from data_loaders import *
import models
from models import *
from utils import *
from models.WideResNet import BasicBlock
from models.ResNet import BasicResNetBlock
from models.layers import LIFSpike

parser = argparse.ArgumentParser(description='Supplementary Code for Adversarially Robust ANN-to-SNN Conversion')
parser.add_argument('--data_dir', default='/DATA_DIR/', type=str, help='dataset directory')
parser.add_argument('--workers', '-j', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--gpu', default='0', type=str, help='device')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('--suffix', default='', type=str, help='suffix')
parser.add_argument('--load_weights', type=str, help='ann statedict name to load weights')
parser.add_argument('--scaling_factor', default=0.3, type=float, help='scaling factor for v_th at reduced timesteps')
parser.add_argument('--soft_reset', action='store_true', help='use soft reset after firing')
parser.add_argument('--use_bias', action='store_true', help='use bias terms in linear layers')
parser.add_argument('--learn_vth', action='store_true', help='perform v_th optimization')
parser.add_argument('--surrogate', default='PCW', type=str, help='surrogate gradient')
parser.add_argument('--gamma', default=1.0, type=float, help='surrogate gradient gamma')
parser.add_argument('--batch_size', '-b', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--optim', default='sgd', type=str, help='adam or sgd')
parser.add_argument('--cutout', action='store_true', help='cutout data augmentation')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--model', default='vgg11_bn', type=str, help='model')
parser.add_argument('--time', '-T', default=8, type=int, metavar='N', help='snn simulation time')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--beta', default=5e-4, type=float, help='weight decay parameter')
parser.add_argument('--trades_beta', default=0., type=float, help='TRADES-loss training weight beta')
parser.add_argument('--mart_beta', default=0., type=float, help='MART-loss training weight beta')
parser.add_argument('--attack', default='', type=str, help='adversarial attack type')
parser.add_argument('--attack_mode', default='', type=str, help='[bptt, bptr, '']')
parser.add_argument('--eps', default=2, type=float, metavar='N', help='attack eps')
parser.add_argument('--alpha', default=0, type=float, metavar='N', help='pgd attack alpha')
parser.add_argument('--steps', default=10, type=int, metavar='N', help='pgd attack steps')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_threshold(model, loader, timesteps, device):
    model.set_simulation_time(T=timesteps)
    thresholds = []

    def wrn_find(layer):
        max_act = 0
        first_done = False
        print('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), data.to(device)
            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if len(output) > 1:  # Customized loop for the BasicBlocks where there are two activations
                    if first_done:
                        if output[1] > max_act:
                            max_act = output[1]
                        if batch_idx == 20:  # use 10 more mini-batches per layer to estimate best thresholds
                            thresholds.append(max_act)
                            print(' {}'.format(thresholds))
                            model.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                            break
                    # Continue setting the first neuron threshold in the BasicBlock first..
                    if output[0] > max_act:
                        max_act = output[0]
                    if batch_idx == 10:
                        thresholds.append(max_act)
                        print(' {}'.format(thresholds))
                        model.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                        first_done = True
                        max_act = 0
                else:
                    output = output[0]
                    if output > max_act:
                        max_act = output
                    if batch_idx == 10:
                        thresholds.append(max_act)
                        print(' {}'.format(thresholds))
                        model.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                        break

    def vgg_find(layer):
        max_act = 0
        print('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), data.to(device)
            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output > max_act:
                    max_act = output
                if batch_idx == 10:
                    thresholds.append(max_act)
                    print(' {}'.format(thresholds))
                    model.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break

    if 'vgg' in args.model.lower():
        for l in model.features.named_children():
            if isinstance(l[1], LIFSpike):
                vgg_find(int(l[0]))

        for c in model.classifier.named_children():
            if isinstance(c[1], LIFSpike):
                vgg_find(len(model.features) + int(c[0]))

    if 'wrn' in args.model.lower():
        for l in model.features.named_children():
            if isinstance(l[1], BasicBlock):
                l[1].find_max_mem = True

        for l in model.features.named_children():
            if int(l[0]) > 1:
                if isinstance(l[1], BasicBlock) or isinstance(l[1], nn.BatchNorm2d) or isinstance(l[1], LIFSpike):
                    wrn_find(int(l[0]))

        for l in model.features.named_children():
            if isinstance(l[1], BasicBlock):
                l[1].find_max_mem = False

    if 'resnet' in args.model.lower():
        for l in model.features.named_children():
            if isinstance(l[1], BasicResNetBlock):
                l[1].find_max_mem = True

        for l in model.features.named_children():
            if isinstance(l[1], LIFSpike):
                wrn_find(int(l[0]))
            else:
                if (isinstance(l[1], BasicResNetBlock) or isinstance(l[1], nn.AdaptiveAvgPool2d)) and (int(l[0]) > 3):
                    wrn_find(int(l[0]))

        for l in model.features.named_children():
            if isinstance(l[1], BasicResNetBlock):
                l[1].find_max_mem = False

    print('\n ANN thresholds: {}'.format(thresholds))
    return thresholds


def main():
    global args
    if args.dataset.lower() == 'cifar10':
        train_dataset, val_dataset, znorm, num_classes = cifar10(args)
    elif args.dataset.lower() == 'cifar100':
        train_dataset, val_dataset, znorm, num_classes = cifar100(args)
    elif args.dataset.lower() == 'svhn':
        train_dataset, val_dataset, znorm, num_classes = svhn(args)
    elif args.dataset.lower() == 'tinyimagenet':
        train_dataset, val_dataset, znorm, num_classes = tinyimagenet(args)
    else:
        raise NotImplementedError

    log_dir = '%s-checkpoints/%s' % (args.dataset, args.model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(log_dir)

    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # Create your model
    model = models.__dict__[args.model.lower()](args.time, num_classes, znorm, args.learn_vth, args.use_bias,
                                                args.soft_reset, args.surrogate, args.gamma)

    if args.load_weights:
        state_dict = torch.load(os.path.join(log_dir, args.load_weights + '.pth'), map_location=torch.device('cpu'))
        load_dict = {}
        for name, param in state_dict.items():
            if not (('num_batches_tracked' in name) or ('running' in name)):
                load_dict[name] = param
        missing_keys, unexpected_keys = model.load_state_dict(load_dict, strict=False)
        print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = None
                m.running_var = None
                m.num_batches_tracked = None

        model.to(device)
        thresholds = find_threshold(model, loader=train_loader, timesteps=100, device=device)
        model.threshold_update(scaling_factor=args.scaling_factor, thresholds=thresholds[:])
        model.set_simulation_time(args.time)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = torch.zeros(m.num_features, device=device)
                m.running_var = torch.ones(m.num_features, device=device)
                m.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
                m.reset_running_stats()

    else:
        model.set_simulation_time(args.time)
        model.to(device)

    if args.attack_mode == 'bptt':
        ff = BPTT_attack
    elif args.attack_mode == 'bptr':
        ff = BPTR_attack
    else:
        ff = None

    step_size = 2.5 * args.eps / args.steps if args.alpha == 0 else args.alpha

    if args.attack.lower() == 'rfgsm':
        adv = attack.RFGSM(model, fwd_function=ff, eps=args.eps / 255, alpha=step_size / 255, loss='kl', T=args.time)
    elif args.attack.lower() == 'pgd':
        adv = attack.PGD(model, fwd_function=ff, eps=args.eps / 255, alpha=step_size / 255, steps=args.steps, T=args.time)
    elif args.attack.lower() == 'tpgd':
        adv = attack.TPGD(model, fwd_function=ff, eps=args.eps / 255, alpha=step_size / 255, steps=args.steps, T=args.time)
    elif args.attack.lower() == 'mart':
        adv = attack.MART(model, fwd_function=ff, eps=args.eps / 255, alpha=step_size / 255, steps=args.steps, T=args.time)
    else:
        adv = None
        assert args.trades_beta == 0.
        assert args.mart_beta == 0.

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.beta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0

    if adv is not None:
        identifier = '%s[%.3f][%s]' % (adv.__class__.__name__, adv.eps, args.attack_mode)
    else:
        identifier = 'clean'

    identifier += '_%s[%.4f]_lr[%.4f]_T%d' % ('wd', args.beta, args.lr, args.time)
    identifier += args.suffix

    logger = get_logger(os.path.join(log_dir, '%s.log' % (identifier)))
    logger.info('start training!')

    if args.load_weights:
        logger.info('\n ANN thresholds: {} \n'.format(thresholds))
        pre_calib_acc = val(model, test_loader, device, args.time)
        logger.info('Pre-calibration Test acc={:.3f}\n'.format(pre_calib_acc))

    for epoch in range(args.epochs):
        loss, acc = train(model, device, train_loader, criterion, optimizer, args.time, adv_train=adv,
                          trades_beta=args.trades_beta, mart_beta=args.mart_beta)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        scheduler.step()
        tmp = val(model, test_loader, device, args.time)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch, args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth' % (identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))


if __name__ == "__main__":
    main()
