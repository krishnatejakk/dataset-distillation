import logging
from contextlib import contextmanager
import torch
import torch.nn as nn
import torchvision
from six import add_metaclass
from torch.nn import init


def init_weights(net, state):
    init_type, init_param = state.init, state.init_param

    if init_type == 'imagenet_pretrained':
        assert net.__class__.__name__ == 'AlexNet'
        state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
        state_dict['classifier.6.weight'] = torch.zeros_like(net.classifier[6].weight)
        state_dict['classifier.6.bias'] = torch.ones_like(net.classifier[6].bias)
        net.load_state_dict(state_dict)
        del state_dict
        return net

    def init_func(m):
        classname = m.__class__.__name__
        if classname.startswith('MetaConv') or classname == 'MetaLinear':
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
            if getattr(m, 'weight', None) is not None:
                if init_type == 'normal':
                    init.normal_(m.weight, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=init_param)
                elif init_type == 'xavier_unif':
                    init.xavier_uniform_(m.weight, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
                elif init_type == 'kaiming_out':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=init_param)
                elif init_type == 'default':
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif 'Norm' in classname:
            if getattr(m, 'weight', None) is not None:
                m.weight.data.fill_(1)
            if getattr(m, 'bias', None) is not None:
                m.bias.data.zero_()

    net.apply(init_func)
    return net


def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)


def clone_tuple(tensors, requires_grad=None):
    return tuple(
        t.detach().clone().requires_grad_(t.requires_grad if requires_grad is None else requires_grad) for t in tensors)