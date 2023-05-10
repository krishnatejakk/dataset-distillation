import logging
import sys
import types
from .networks import *


def get_networks(state, N=None, arch=None):
    N = N or state.local_n_nets
    arch = arch or state.arch
    mod = sys.modules[__name__]
    cls = getattr(mod, arch)
    if state.input_size not in cls.supported_dims:
        raise RuntimeError("{} doesn't support input size {}".format(cls.__name__, state.input_size))
    logging.info('Build {} {} network(s) with [{}({})] init'.format(N, arch, state.init, state.init_param))
    nets = []
    for n in range(N):
        net = cls(state)
        net.reset(state)  # verbose only last one
        nets.append(net)
    return nets


def get_zen_networks(state, N=None, archs=None):
    N = N or state.local_n_nets
    archs = archs or state.zen_archs
    archs = archs.split(',')
    mod = sys.modules[__name__]
    nets = []
    for arch in archs:
        obj = getattr(mod, arch)
        if isinstance(obj, types.FunctionType):
            cls = obj(state)
        else:
            cls = obj
        if state.input_size not in cls.supported_dims:
            raise RuntimeError("{} doesn't support input size {}".format(cls.__name__, state.input_size))
        logging.info('Build {} {} network(s) with [{}({})] init'.format(N, arch, state.init, state.init_param))
        for n in range(N):
            if isinstance(obj, types.FunctionType):
                net = cls
            else:
                net = cls(state)
            net.reset(state)  # verbose only last one
            nets.append(net)
    return nets