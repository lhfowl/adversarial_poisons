"""Base victim class."""

import torch

from .models import get_model
from .training import get_optimizers, run_step
from .optimization_strategy import training_strategy
from ..utils import average_dicts
from ..consts import BENCHMARK, SHARING_STRATEGY
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class _ClientBase:
    """Implement model-specific code and behavior.

    Expose:
    Attributes:
     - model
     - optimizer
     - scheduler
     - criterion

     Methods:
     - initialize
     - train
     - retrain
     - validate
     - iterate

     - compute
     - gradient
     - eval

     Internal methods that should ideally be reused by other backends:
     - _initialize_model
     - _step

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        if self.args.ensemble < len(self.args.net):
            raise ValueError(f'More models requested than ensemble size.'
                             f'Increase ensemble size or reduce models.')
        if self.args.ensemble > 1:
            self.initialize()
        else:
            self.initialize()

    def gradient(self, images, labels):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        raise NotImplementedError()
        return grad, grad_norm

    def compute(self, function):
        """Compute function on all models.

        Function has arguments: model, criterion
        """
        raise NotImplementedError()

    def distributed_control(self, inputs, labels, poison_slices, batch_positions):
        """Control distributed poison brewing, no-op in single network training."""
        randgen = None
        return inputs, labels, poison_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable. No-op for single network training."""
        return input

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        raise NotImplementedError()


    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        raise NotImplementedError()

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def train(self, furnace, max_epoch=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
        print('Starting clean training ...')
        '''
        if self.args.feat_extractor_crafting:
            defs = training_strategy(self.args.net[0], self.args)
            self.optimizer = torch.optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)
        '''
        return self._iterate(furnace, poison_delta=None, max_epoch=max_epoch)

    def eval(self, dropout=True):
        """Switch everything into evaluation mode."""
        raise NotImplementedError()

    def _iterate(self, furnace, poison_delta):
        """Validate a given poison by training the model and checking target accuracy."""
        raise NotImplementedError()

    def _adversarial_step(self, furnace, poison_delta, step, poison_targets, true_classes):
        """Step through a model epoch to in turn minimize target loss."""
        raise NotImplementedError()

    def _initialize_model(self, model_name, feature_extractor=False):

        model = get_model(model_name, self.args.dataset, pretrained=self.args.pretrained)
        # Define training routine
        defs = training_strategy(model_name, self.args)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizers(model, self.args, defs)

        return model, defs, criterion, optimizer, scheduler


    def _step(self, furnace, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler):
        """Single epoch. Can't say I'm a fan of this interface, but ..."""
        run_step(furnace, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler)
