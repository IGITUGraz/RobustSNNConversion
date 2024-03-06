import torch
import torch.nn as nn
from torchattacks.attack import Attack


class FGSM(Attack):
    def __init__(self, model, fwd_function=None, eps=8/255, T=None, surrogate='PCW', gamma=1.0):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.forward_function = fwd_function
        self.surrogate = surrogate
        self.gamma = gamma
        self.T = T
        print('FGSM attack with epsilon: ', eps)
        if T > 0:
            print('Surrogate: ', surrogate, ' and gamma: ', gamma)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        if self.forward_function is not None:
            outputs = self.forward_function(self.model, images, self.T, self.surrogate, self.gamma)
        else:
            outputs = self.model(images)

        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images
