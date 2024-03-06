import torch
import torch.nn as nn
from torchattacks.attack import Attack
from attack import *


class Ensemble(Attack):
    def __init__(self, model, fwd_functions=None, eps=8/255, alpha=2/255, steps=10, T=None, version='autoattack',
                 seed=0, verbose=False, n_classes=10):
        super().__init__("Ensemble", model)
        if len(fwd_functions) == 1:
            self.ff = fwd_functions[0]
        else:
            self.ff_1, self.ff_2, self.ff_3 = fwd_functions
        self.T = T
        self.eps = eps
        self.verbose = verbose
        self._supported_mode = ['default']
        print('Ensemble attack with epsilon: ', eps)

        if version == 'autoattack':  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            self._multiattack = MultiAttack([
                APGD(model, fwd_function=self.ff, T=T, eps=eps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGDT(model, fwd_function=self.ff, T=T, eps=eps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, fwd_function=self.ff, T=T, eps=eps, seed=seed, verbose=verbose, multi_targeted=True, n_classes=n_classes, n_restarts=1),
                Square(model, fwd_function=self.ff, T=T, eps=eps, seed=seed, verbose=verbose, n_queries=5000, n_restarts=1),
            ], fwd_function=self.ff, T=T)

        elif version == 'apgd-dlr':
            self._multiattack = MultiAttack([
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=3.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.25, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.25, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=4.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='STE', eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_2, T=T, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
                APGD(model, fwd_function=self.ff_3, T=T, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='dlr', n_restarts=1),
            ], fwd_function=self.ff_1, T=T)

        elif version == 'apgdt':
            self._multiattack = MultiAttack([
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=3.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.25, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.25, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=4.0, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_1, T=T, surrogate='STE', eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_2, T=T, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                APGDT(model, fwd_function=self.ff_3, T=T, eps=eps, steps=steps, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
            ], fwd_function=self.ff_1, T=T)

        elif version == 'apgd':
            self._multiattack = MultiAttack([
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=3.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.25, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.25, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.5, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=1.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=2.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=4.0, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_1, T=T, surrogate='STE', eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_2, T=T, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGD(model, fwd_function=self.ff_3, T=T, eps=eps, steps=steps, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
            ], fwd_function=self.ff_1, T=T)

        elif version == 'pgd':
            self._multiattack = MultiAttack([
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=1.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=2.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=3.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.5, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.25, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=0.5, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=1.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=2.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=0.5, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=1.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=2.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.25, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.5, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=1.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=2.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=4.0, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_1, T=T, surrogate='STE', eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_2, T=T, eps=eps, alpha=alpha, steps=steps),
                PGD(model, fwd_function=self.ff_3, T=T, eps=eps, alpha=alpha, steps=steps),
            ], fwd_function=self.ff_1, T=T)

        elif version == 'rfgsm':
            self._multiattack = MultiAttack([
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=1.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=2.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=3.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.5, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.25, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=0.5, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=1.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=2.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=0.5, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=1.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=2.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.25, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.5, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=1.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=2.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=4.0, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_1, T=T, surrogate='STE', eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_2, T=T, eps=eps, alpha=alpha, loss='ce'),
                RFGSM(model, fwd_function=self.ff_3, T=T, eps=eps, alpha=alpha, loss='ce'),
            ], fwd_function=self.ff_1, T=T)

        elif version == 'fgsm':
            self._multiattack = MultiAttack([
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=1.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=2.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=3.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.5, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='PCW', gamma=0.25, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=0.5, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=1.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP-D', gamma=2.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=0.5, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=1.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='EXP', gamma=2.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.25, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=0.5, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=1.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=2.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='RECT', gamma=4.0, eps=eps),
                FGSM(model, fwd_function=self.ff_1, T=T, surrogate='STE', eps=eps),
                FGSM(model, fwd_function=self.ff_2, T=T, eps=eps),
                FGSM(model, fwd_function=self.ff_3, T=T, eps=eps),
            ], fwd_function=self.ff_1, T=T)

        else:
            raise ValueError("Not a valid version.")

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._multiattack(images, labels)
        return adv_images


class MultiAttack(Attack):
    def __init__(self, attacks, fwd_function=None, T=None, surrogate='PCW', gamma=1.0, verbose=False):
        super().__init__("MultiAttack", attacks[0].model)
        self.forward_function = fwd_function
        self.surrogate = surrogate
        self.gamma = gamma
        self.T = T
        self.attacks = attacks
        self.verbose = verbose
        self.supported_mode = ['default']

        self.check_validity()

        self._accumulate_multi_atk_records = False
        self._multi_atk_records = [0.0]

    def check_validity(self):
        if len(self.attacks) < 2:
            raise ValueError("More than two attacks should be given.")

        ids = [id(attack.model) for attack in self.attacks]
        if len(set(ids)) != 1:
            raise ValueError("At least one of attacks is referencing a different model.")

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        batch_size = images.shape[0]
        fails = torch.arange(batch_size).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        multi_atk_records = [batch_size]

        for _, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])

            if self.forward_function is not None:
                outputs = self.forward_function(self.model, adv_images, self.T, self.surrogate, self.gamma)
            else:
                outputs = self.model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = (pre == labels[fails])
            wrongs = ~corrects

            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(torch.arange(fails.shape[0]).to(self.device), wrongs)

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, corrects)
            multi_atk_records.append(len(fails))

            if len(fails) == 0:
                break

        if self.verbose:
            print(self._return_sr_record(multi_atk_records))

        if self._accumulate_multi_atk_records:
            self._update_multi_atk_records(multi_atk_records)

        return final_images

    def _clear_multi_atk_records(self):
        self._multi_atk_records = [0.0]

    def _covert_to_success_rates(self, multi_atk_records):
        sr = [((1-multi_atk_records[i]/multi_atk_records[0])*100) for i in range(1, len(multi_atk_records))]
        return sr

    def _return_sr_record(self, multi_atk_records):
        sr = self._covert_to_success_rates(multi_atk_records)
        return "Attack success rate: "+" | ".join(["%2.2f %%"%item for item in sr])

    def _update_multi_atk_records(self, multi_atk_records):
        for i, item in enumerate(multi_atk_records):
            self._multi_atk_records[i] += item

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False,
             save_predictions=False, save_clean_images=False):
        r"""
        Overridden.
        """
        self._clear_multi_atk_records()
        prev_verbose = self.verbose
        self.verbose = False
        self._accumulate_multi_atk_records = True

        for i, attack in enumerate(self.attacks):
            self._multi_atk_records.append(0.0)

        if return_verbose:
            rob_acc, l2, elapsed_time = super().save(data_loader, save_path,
                                                     verbose, return_verbose,
                                                     save_predictions,
                                                     save_clean_images)
            sr = self._covert_to_success_rates(self._multi_atk_records)
        elif verbose:
            super().save(data_loader, save_path, verbose,
                         return_verbose, save_predictions,
                         save_clean_images)
            sr = self._covert_to_success_rates(self._multi_atk_records)
        else:
            super().save(data_loader, save_path, False,
                         False, save_predictions,
                         save_clean_images)

        self._clear_multi_atk_records()
        self._accumulate_multi_atk_records = False
        self.verbose = prev_verbose

        if return_verbose:
            return rob_acc, sr, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        r"""
        Overridden.
        """
        print("- Save progress: %2.2f %% / Robust accuracy: %2.2f %%"%(progress, rob_acc)+\
              " / "+self._return_sr_record(self._multi_atk_records)+\
              ' / L2: %1.5f (%2.3f it/s) \t'%(l2, elapsed_time), end=end)