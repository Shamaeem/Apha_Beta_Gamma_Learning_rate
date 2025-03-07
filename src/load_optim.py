"""
Load the desired optimizer.
"""

import torch.optim as optim
from sgd_lr_decay import SGDLRDecay
from sls import Sls
from scgWSA import SCGWSA
from scgWSA2 import SCGWSA2
from scgs import SCGS
from scgs2 import SCGS2

def load_optim(params, optim_method, eta0, alpha, c, milestones, T_max, 
               n_batches_per_epoch, nesterov, momentum, weight_decay, gamma, coeff):
    """
    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        optim_method: which optimizer to use.
        eta0: starting step size.
        alpha: decaying factor for various methods.
        c: used in line search.
        milestones: used for SGD stage decay denoting when to decrease the
            step size, unit in iteration.
        T_max: total number of steps.
        n_batches_per_epoch: number of batches in one train epoch.
        nesterov: whether to use nesterov momentum (True) or not (False).
        momentum: momentum factor used in variants of SGD.
        weight_decay: weight decay factor.

    Outputs:
        an optimizer
    """

    if optim_method == 'SGD' or optim_method == 'SGD_ReduceLROnPlateau':
        optimizer = optim.SGD(params=params, lr=eta0, momentum=momentum,
                              weight_decay=weight_decay, nesterov=nesterov)
    elif optim_method == 'Adam':
        optimizer = optim.Adam(params=params, lr=eta0,
                               weight_decay=weight_decay)
    elif optim_method.startswith('SGD') and optim_method.endswith('Decay'):
        if optim_method == 'SGD_Exp_Decay':
            scheme = 'exp'
        elif optim_method == 'SGD_1t_Decay':
            scheme = '1t'
        elif optim_method == 'SGD_1sqrt_Decay':
            scheme = '1sqrt'
        elif optim_method == 'SGD_1sqrtlnt_Decay':
            scheme = '1sqrtlnt'
        elif optim_method == 'SGD_Stage_Decay':
            scheme = 'stage'
        elif optim_method == 'SGD_Cosine_Decay':
            scheme = 'cosine'
        elif optim_method == 'SGD_tan1_Decay':
            scheme = 'tan1'
        elif optim_method == 'SGD_tan2_Decay':
            scheme = 'tan2'
        elif optim_method == 'SGD_tan3_Decay':
            scheme = 'tan3'
        optimizer = SGDLRDecay(params=params, scheme=scheme, eta0=eta0,
                               alpha=alpha, milestones=milestones, T_max=T_max,
                               momentum=momentum, weight_decay=weight_decay,
                               nesterov=nesterov, gamma=gamma, coeff=coeff)
    elif optim_method == 'SLS-Armijo0':
        optimizer = Sls(params=params, n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=eta0, c=c, reset_option=0,
                        line_search_fn="armijo")
    elif optim_method == 'SLS-Armijo1':
        optimizer = Sls(params=params, n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=eta0, c=c, reset_option=1,
                        line_search_fn="armijo")
    elif optim_method == 'SLS-Armijo2':
        optimizer = Sls(params=params, n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=eta0, c=c, reset_option=2,
                        line_search_fn="armijo")
    elif optim_method == 'SCGWSA':
        optimizer = SCGWSA(params=params)
    elif optim_method == 'SCGWSA2':
        optimizer = SCGWSA2(params=params)
    elif optim_method == 'SCGS':
        optimizer = SCGS(params=params)
    elif optim_method == 'SCGS2':
        optimizer = SCGS2(params=params, eta0=eta0, T_max=T_max)
    
    else:
        raise ValueError("Invalid optimizer: {}".format(optim_method))

    return optimizer
