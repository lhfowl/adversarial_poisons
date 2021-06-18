"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss
import pdb
import random
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterTargeted(_Forgemaster):

    def _define_objective(self, inputs, labels, targets):
        """Implement the closure here."""
        def closure(model, criterion, optimizer, target_grad, target_clean_grad):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            outputs = model(inputs)
            new_labels = self._label_map(outputs, labels)
            loss = criterion(outputs, new_labels)
            loss.backward(retain_graph=self.retain)
            prediction = (outputs.data.argmax(dim=1) == new_labels).sum()
            #max_class = outputs.data.argmax(dim=1)
            #softmax = torch.nn.functional.softmax(outputs, dim=1)[torch.arange(len(labels)), max_class]
            return loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _label_map(self, outputs, labels):
        # This is a naiive permutation on the label space. You can implement
        # any permutation you like here.
        new_labels = (labels + 1) % outputs.shape[1]
        return new_labels
