#!/bin/bash

dir="/DATA_DIR/"

# Example scripts to adversarially train baseline ANNs
python -u main_train.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 0 --cutout --attack 'pgd' --eps 2
python -u main_train.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 0 --cutout --attack 'tpgd' --trades_beta 6. --eps 2
python -u main_train.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 0 --cutout --attack 'mart' --mart_beta 4. --eps 2

# Example scripts for adversarially robust ANN-to-SNN conversion
python -u main_train.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 8 --beta 0.001 --lr 0.001 --learn_vth --use_bias --attack 'rfgsm' --attack_mode 'bptt' --eps 2 --alpha 1 --load_weights 'PGD[0.008][]_wd[0.0005]_lr[0.1000]_T0' --epochs 60 --trades_beta 2. --suffix '_CONV_PGD[0.008]'
python -u main_train.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 8 --beta 0.001 --lr 0.001 --learn_vth --use_bias --attack 'rfgsm' --attack_mode 'bptt' --eps 2 --alpha 1 --load_weights 'TPGD[0.008][]_wd[0.0005]_lr[0.1000]_T0' --epochs 60 --trades_beta 2. --suffix '_CONV_TPGD[0.008]'
python -u main_train.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 8 --beta 0.001 --lr 0.001 --learn_vth --use_bias --attack 'rfgsm' --attack_mode 'bptt' --eps 2 --alpha 1 --load_weights 'MART[0.008][]_wd[0.0005]_lr[0.1000]_T0' --epochs 60 --trades_beta 2. --suffix '_CONV_MART[0.008]'

# Example scripts for evaluating converted SNNs
python -u main_test.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 8 --identifier 'RFGSM[0.008][bptt]_wd[0.0010]_lr[0.0010]_T8_CONV_PGD[0.008]' --learn_vth --use_bias --eps 8 --attack ensemble --ens_version fgsm --attack_mode bptt
python -u main_test.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 8 --identifier 'RFGSM[0.008][bptt]_wd[0.0010]_lr[0.0010]_T8_CONV_TPGD[0.008]' --learn_vth --use_bias --eps 8 --attack ensemble --ens_version pgd --attack_mode bptt --steps 20
python -u main_test.py --data_dir $dir --dataset cifar10 --model vgg11_bn -T 8 --identifier 'RFGSM[0.008][bptt]_wd[0.0010]_lr[0.0010]_T8_CONV_MART[0.008]' --learn_vth --use_bias --eps 8 --attack ensemble --ens_version apgd --attack_mode bptt
