"""Implement client behavior, for single-client, ensemble and stuff."""
import torch

from .client_distributed import _ClientDistributed
from .client_ensemble import _ClientEnsemble
from .client_single import _ClientSingle

def Client(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.local_rank is not None:
        return _ClientDistributed(args, setup)
    elif args.ensemble == 1:
        return _ClientSingle(args, setup)
    elif args.ensemble > 1:
        return _ClientEnsemble(args, setup)


from .optimization_strategy import training_strategy
__all__ = ['Client', 'training_strategy']
