import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack


class MART(Attack):
    def __init__(self, model, fwd_function=None, eps=8/255, alpha=2/255, steps=10, T=None, surrogate='PCW', gamma=1.0):
        super().__init__("MART", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self._supported_mode = ['default']
        self.forward_function = fwd_function
        self.surrogate = surrogate
        self.gamma = gamma
        self.T = T
        print('MART-PGD attack with epsilon: ', eps, ' and step size: ', alpha)
        if T > 0:
            print('Surrogate: ', surrogate, ' and gamma: ', gamma)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        adv_images = adv_images + 0.001 * torch.randn_like(adv_images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            if self.forward_function is not None:
                outputs = self.forward_function(self.model, adv_images, self.T, self.surrogate, self.gamma)
            else:
                outputs = self.model(adv_images)

            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
