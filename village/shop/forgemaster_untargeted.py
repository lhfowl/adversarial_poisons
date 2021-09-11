"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss, reverse_xent_avg
import pdb
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterUntargeted(_Forgemaster):
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
            loss = -criterion(outputs,labels)
            loss.backward(retain_graph=self.retain)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            return loss.detach().cpu(), prediction.detach().cpu()
        return closure
