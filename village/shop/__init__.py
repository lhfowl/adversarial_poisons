"""Interface for poison recipes."""
from .forgemaster_matching import ForgemasterGradientMatching, ForgemasterGradientMatchingNoisy
from .forgemaster_metapoison import ForgemasterMetaPoison
from .forgemaster_poison_frogs import ForgemasterFrogs
from .forgemaster_untargeted import ForgemasterUntargeted
from .forgemaster_targeted import ForgemasterTargeted
from .forgemaster_explosion import ForgemasterExplosion
from .forgemaster_tensorclog import ForgemasterTensorclog

import torch


def Forgemaster(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'gradient-matching':
        return ForgemasterGradientMatching(args, setup)
    elif args.recipe == 'gradient-matching-private':
        return ForgemasterGradientMatchingNoisy(args, setup)
    elif args.recipe == 'metapoison':
        return ForgemasterMetaPoison(args, setup)
    elif args.recipe == 'poison-frogs':
        return ForgemasterFrogs(args, setup)
    elif args.recipe == 'grad_explosion':
        return ForgemasterExplosion(args, setup)
    elif args.recipe == 'tensorclog':
        return ForgemasterTensorclog(args, setup)
    elif args.recipe == 'untargeted':
        return ForgemasterUntargeted(args, setup)
    elif args.recipe == 'targeted':
        return ForgemasterTargeted(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Forgemaster']
