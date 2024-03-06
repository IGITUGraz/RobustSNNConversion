import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack


class RFGSM(Attack):
    def __init__(self, model, fwd_function=None, T=None, surrogate='PCW', gamma=1.0, eps=8/255, alpha=4/255, loss='kl'):
        super().__init__("RFGSM", model)
        self.forward_function = fwd_function
        self.surrogate = surrogate
        self.gamma = gamma
        self.T = T
        self.eps = eps
        self.alpha = alpha
        self.loss = loss
        self.supported_mode = ['default']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        if self.loss == 'kl':
            if self.forward_function is not None:
                outputs = self.forward_function(self.model, images, self.T, self.surrogate, self.gamma).detach()
            else:
                outputs = self.model(images).detach()

            criterion_kl = nn.KLDivLoss(size_average=False)
            adv_images = images + self.alpha * torch.randn_like(images).sign()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            adv_images.requires_grad = True
            if self.forward_function is not None:
                outputs_adv = self.forward_function(self.model, adv_images, self.T, self.surrogate, self.gamma)
            else:
                outputs_adv = self.model(adv_images)
            cost = criterion_kl(F.log_softmax(outputs_adv, dim=1), F.softmax(outputs, dim=1))

        else:   # loss = 'ce'
            labels = labels.clone().detach().to(self.device)

            loss = nn.CrossEntropyLoss()
            adv_images = images + self.alpha*torch.randn_like(images).sign()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            adv_images.requires_grad = True
            if self.forward_function is not None:
                outputs = self.forward_function(self.model, adv_images, self.T, self.surrogate, self.gamma)
            else:
                outputs = self.model(adv_images)
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + (self.eps - self.alpha) * grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images
