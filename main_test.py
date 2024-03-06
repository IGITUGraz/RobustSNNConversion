import argparse
import os
import random
import sys
from utils import *
from data_loaders import *
from torchvision import datasets
import numpy as np
import models
import attack
import copy
import torch

parser = argparse.ArgumentParser(description='Supplementary Code for Adversarially Robust ANN-to-SNN Conversion')
parser.add_argument('--data_dir', default='/DATA_DIR/', type=str, help='dataset directory')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-sd', '--seed', default=42, type=int, help='seed for initializing training.')
parser.add_argument('--gpu', default='0', type=str, help='device')
parser.add_argument('-suffix', '--suffix', default='', type=str, help='suffix')
parser.add_argument('--cutout', action='store_true', help='cutout data augmentation')
parser.add_argument('-data', '--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('-arch', '--model', default='vgg11', type=str, help='model')
parser.add_argument('-T', '--time', default=8, type=int, metavar='N', help='snn simulation time')
parser.add_argument('-id', '--identifier', type=str, help='model statedict identifier to load')
parser.add_argument('--surrogate', default='PCW', type=str, help='surrogate gradient')
parser.add_argument('--gamma', default=1.0, type=float, help='surrogate gradient gamma')
parser.add_argument('--learn_vth', action='store_true', help='perform v_th optimization')
parser.add_argument('--use_bias', action='store_true', help='use bias terms in linear layers')
parser.add_argument('--soft_reset', action='store_true', help='use soft reset after firing')
parser.add_argument('--attack', default='', type=str, help='adversarial attack type')
parser.add_argument('--attack_mode', default='', type=str, help='[bptt, bptr, '']')
parser.add_argument('--eps', default=8, type=float, metavar='N', help='attack eps')
parser.add_argument('--alpha', default=0, type=float, metavar='N', help='pgd attack alpha')
parser.add_argument('--steps', default=10, type=int, metavar='N', help='pgd attack steps')
parser.add_argument('--ens_version', default='autoattack', type=str, help='ensemble attack type')
parser.add_argument('--n_queries', default=5000, type=int, help='number of queries for square attack')
parser.add_argument('--bbmodel', default='', type=str, help='black box model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args

    model_dir = '%s-checkpoints/%s' % (args.dataset, args.model)
    log_dir = '%s-results/%s' % (args.dataset, args.model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(model_dir)

    logger = get_logger(os.path.join(log_dir, '%s.log' % (args.identifier + args.suffix)))
    logger.info('start testing!')

    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.dataset.lower() == 'cifar10':
        _, val_dataset, znorm, num_classes = cifar10(args)
    elif args.dataset.lower() == 'cifar100':
        _, val_dataset, znorm, num_classes = cifar100(args)
    elif args.dataset.lower() == 'svhn':
        _, val_dataset, znorm, num_classes = svhn(args)
    elif args.dataset.lower() == 'tinyimagenet':
        _, val_dataset, znorm, num_classes = tinyimagenet(args)
    else:
        raise NotImplementedError

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    # Create your model
    model = models.__dict__[args.model.lower()](args.time, num_classes, znorm, args.learn_vth, args.use_bias,
                                                args.soft_reset, args.surrogate, args.gamma)
    model.set_simulation_time(args.time)
    model.to(device)

    # have bb model
    if len(args.bbmodel) > 0:
        bbmodel = copy.deepcopy(model)
        bbstate_dict = torch.load(os.path.join(model_dir, args.bbmodel + '.pth'), map_location=torch.device('cpu'))
        if (args.time != 0) and ('_T0_' in args.bbmodel):
            print('Loaded black-box ANN transfer attack model:')
            bbmodel.set_simulation_time(0)
            bbmodel.load_state_dict(bbstate_dict, strict=False)
            bbmodel.set_simulation_time(0)
            acc = val(bbmodel, test_loader, device, 0)
            logger.info('Black-box model accuracy: ={:.3f}'.format(acc))
        else:
            print('Loaded black-box SNN transfer attack model:')
            bbmodel.load_state_dict(bbstate_dict, strict=True)
            acc = val(bbmodel, test_loader, device, args.time)
            logger.info('Black-box model accuracy: ={:.3f}'.format(acc))
        print(args.bbmodel)
    else:
        bbmodel = None

    if len(args.bbmodel) > 0:
        print('Evaluating as a black-box transfer attack...')
        atkmodel = bbmodel
    else:
        atkmodel = model

    if args.attack_mode == 'bptt':
        ff = BPTT_attack
    elif args.attack_mode == 'bptr':
        ff = BPTR_attack
    elif args.attack_mode == 'none':
        ff = None
    else:
        ff = Act_attack

    step_size = 2.5 * args.eps / args.steps if args.alpha == 0 else args.alpha

    if args.attack.lower() == 'fgsm':
        atk = attack.FGSM(atkmodel, fwd_function=ff, eps=args.eps/255, T=args.time, surrogate=args.surrogate, gamma=args.gamma)
    elif args.attack.lower() == 'rfgsm':
        atk = attack.RFGSM(atkmodel, fwd_function=ff, eps=args.eps/255, alpha=step_size/255, T=args.time, surrogate=args.surrogate, gamma=args.gamma)
    elif args.attack.lower() == 'pgd':
        atk = attack.PGD(atkmodel, fwd_function=ff, eps=args.eps/255, alpha=step_size/255, steps=args.steps, T=args.time, surrogate=args.surrogate, gamma=args.gamma)
    elif args.attack.lower() == 'apgd':
        atk = attack.APGD(atkmodel, fwd_function=ff, eps=args.eps/255, T=args.time, surrogate=args.surrogate, gamma=args.gamma)
    elif args.attack.lower() == 'square':
        atk = attack.Square(atkmodel, fwd_function=ff, eps=args.eps/255, T=args.time, n_queries=args.n_queries)
    elif args.attack.lower() == 'ensemble':
        if args.ens_version == 'autoattack':
            atk = attack.Ensemble(atkmodel, fwd_functions=[ff], eps=args.eps/255, T=args.time, n_classes=num_classes)
        else:
            atk = attack.Ensemble(atkmodel, fwd_functions=[BPTT_attack, BPTR_attack, Act_attack], T=args.time, eps=args.eps/255, alpha=step_size/255, steps=args.steps, version=args.ens_version)
    else:
        atk = None

    state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)
    acc = val(model, test_loader, device, args.time, atk)
    logger.info('Attack Test acc={:.3f}'.format(acc))


if __name__ == "__main__":
    main()
