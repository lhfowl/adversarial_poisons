"""Implement client behavior, for single-client, ensemble and stuff."""
import torch

from .client_single import _ClientSingle

def Client(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.ensemble == 1:
        return _ClientSingle(args, setup)
    else:
        raise NotImplementedError()


from .optimization_strategy import training_strategy
__all__ = ['Client', 'training_strategy']
