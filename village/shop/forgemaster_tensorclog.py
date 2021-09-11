"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss, reverse_xent
import pdb
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterTensorclog(_Forgemaster):
    """Brew passenger poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels):
        """Implement the closure here."""
        def closure(model, criterion, optimizer):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            outputs = model(inputs)
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                criterion = cw_loss
            else:
                pass  # use the default for untargeted or targeted cross entropy
            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)

            explosion_loss = 0
            for grad in poison_grad:
                explosion_loss += grad.pow(2).sum()
            explosion_loss = explosion_loss.sqrt()
            explosion_loss.backward(retain_graph=self.retain)
            return explosion_loss.detach().cpu(), prediction.detach().cpu()
        return closure
