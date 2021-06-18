"""Single model default victim class."""

import torch
import numpy as np
from collections import defaultdict


from ..utils import set_random_seed
from ..consts import BENCHMARK
import pdb
torch.backends.cudnn.benchmark = BENCHMARK

from .client_base import _ClientBase


class _ClientSingle(_ClientBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation.

    """

    """ Methods to initialize a model."""

    def initialize(self, seed=None, feature_extractor=False):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.criterion, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], feature_extractor=feature_extractor)

        self.model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking target accuracy."""
        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs

        def loss_fn(model, outputs, labels):
            return self.criterion(outputs, labels)

        single_setup = (self.model, self.defs, self.criterion, self.optimizer, self.scheduler)
        for self.epoch in range(max_epoch):
            self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup)
            if self.args.dryrun:
                break
        return stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally: minimize target loss."""
        stats = defaultdict(list)

        def loss_fn(model, outputs, labels):
            normal_loss = self.criterion(outputs, labels)
            model.eval()
            if self.args.adversarial != 0:
                target_loss = 1 / self.defs.batch_size * self.criterion(model(poison_targets), true_classes)
            else:
                target_loss = 0
            model.train()
            return normal_loss + self.args.adversarial * target_loss

        single_setup = (self.model, self.criterion, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            print('Model reset to epoch 0.')
            self.model, self.criterion, self.optimizer, self.scheduler = self._initialize_model()
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model()

    def gradient(self, images, labels, criterion=None, batched=False):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        if not batched:
            if criterion is None:
                loss = self.criterion(self.model(images), labels)
            else:
                loss = criterion(self.model(images), labels)
            gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        else:
            for i in range(int(len(images)/100)):
                print(f'Iteration of calculating poison grad: {i}')
                loss = criterion(self.model(images[i*100:(i+1)*100]), labels[i*100:(i+1)*100])
                #temp_gradients.append(torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
                temp_gradient = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
                if i == 0:
                    gradients = list(temp_gradient)
                else:
                    for j in range(len(gradients)):
                        gradients[j] += temp_gradient[j]
                        if i == int(len(images)/100) - 1:
                            gradients[j] /= int(len(images)/100)

    def batched_gradient(self, kettle, setup, criterion):
        targetloader = torch.utils.data.DataLoader(kettle.targetset, batch_size=min(kettle.batch_size, len(kettle.targetset)),
                                                       shuffle=False, drop_last=False)

        for idx, (inputs, targets, _) in enumerate(targetloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=self.setup['device'], dtype=torch.long)
            loss = criterion(self.model(inputs), targets)
            temp_gradient = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
            if idx == 0:
                gradients = list(temp_gradient)
            else:
                for j in range(len(gradients)):
                    gradients[j] += temp_gradient[j]
                    if idx == len(targetloader) - 1:
                        gradients[j] /= len(targetloader)

        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.criterion, self.optimizer, *args)
