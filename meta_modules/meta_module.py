import logging
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
from .utils import init_weights
from ..networks.utils import PatchModules, ReparamModule
from six import add_metaclass
# from weight_batch_norm import BatchWeightNorm2d, BatchWeightNorm1d, BatchNorm1d

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def build_model():
    model = WideResNet(depth=10, num_classes=10, widen_factor=2, dropRate=0.0)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.params()])))
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    return model

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)
        

@add_metaclass(PatchModules)
class CombinedModule(MetaModule, ReparamModule):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        for name, param in self.named_leaves():
            if name == 'weight':
                init.kaiming_uniform_(param, a=math.sqrt(5))
            elif name == 'bias' and param is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(param)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(param, -bound, bound)

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            new_flat_w = self.get_param(clone=True)
            with self.unflatten_weight(new_flat_w):
                super().update_params(lr_inner, first_order, source_params, detach)
                self.register_parameter('flat_w', nn.Parameter(new_flat_w, requires_grad=True))
        else:
            super().update_params(lr_inner, first_order, source_params, detach)

    @contextmanager
    def unflatten_weight(self, flat_w):
        with ReparamModule.unflatten_weight(self, flat_w) as _:
            yield


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats
        self.update_batch_stats = True

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        if self.update_batch_stats:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)
        else:
            return F.batch_norm(x, None, None, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats
        self.update_batch_stats = True

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        if self.update_batch_stats:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)
        else:
            return F.batch_norm(x, None, None, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

def relu():
    return nn.LeakyReLU(0.1)

class MetaBasicBlock(MetaModule):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(MetaBasicBlock, self).__init__()

        self.bn1 = MetaBatchNorm2d(in_planes)
        self.relu1 = relu()
        self.conv1 = MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(out_planes)
        self.relu2 = relu()
        self.conv2 = MetaConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                 padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class MetaNetworkBlock(MetaModule):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(MetaNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(MetaModule):
    def __init__(self, depth, num_classes, widen_factor=1, transform_fn=None, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MetaBasicBlock
        # 1st conv before any network block
        self.conv1 = MetaConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False) # for mnist
        # 1st block
        self.block1 = MetaNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = MetaNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = MetaNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = MetaBatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = MetaLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.transform_fn = transform_fn

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaLinear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 7)
        # out = nn.AdaptiveAvgPool2d(1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


class VNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class VNet2(nn.Module):
    def __init__(self):
        super(VNet2, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, 6, 5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(6, 16, 5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(16, 120, 5))
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(84, 1))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        x = self.fc_layers(x).squeeze()
        return F.sigmoid(x)

class FFNN(MetaModule):
    """
    Feed-Forward Neural Network (FFNN) for MNIST.
    Total 4 hidden layers are used as 28*28 -> (1200, 600, 300, 150) -> 10.
    We apply batchnorm and ReLU.
    We add isotropic noise to every hidden layer to stablize training.
    """
    def __init__(self, params):
        super(FFNN, self).__init__()
        self.fc1 = MetaLinear(28 * 28, 1200)
        self.fc2 = MetaLinear(1200, 600)
        self.fc3 = MetaLinear(600, 300)
        self.fc4 = MetaLinear(300, 150)
        self.fc5 = MetaLinear(150, 10)
        self.bn1 = MetaBatchNorm2d(1200)
        self.bn2 = MetaBatchNorm2d(600)
        self.bn3 = MetaBatchNorm2d(300)
        self.bn4 = MetaBatchNorm2d(150)

    def forward(self, X):
        out = X.view(X.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn2(self.fc2(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn3(self.fc3(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn4(self.fc4(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = self.fc5(out)
        return out

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


class FFNN6(MetaModule):
    """
    Feed-Forward Neural Network (FFNN) for MNIST.
    Total 4 hidden layers are used as 28*28 -> (1200, 600, 300, 150) -> 10.
    We apply batchnorm and ReLU.
    We add isotropic noise to every hidden layer to stablize training.
    """
    def __init__(self, params):
        super(FFNN6, self).__init__()
        self.fc1 = MetaLinear(28 * 28, 1200)
        self.fc2 = MetaLinear(1200, 600)
        self.fc3 = MetaLinear(600, 300)
        self.fc4 = MetaLinear(300, 150)
        self.fc5 = MetaLinear(150, 10)
        self.bn1 = MetaBatchNorm1d(1200)
        self.bn2 = MetaBatchNorm1d(600)
        self.bn3 = MetaBatchNorm1d(300)
        self.bn4 = MetaBatchNorm1d(150)

    def forward(self, X):
        out = X.view(X.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn2(self.fc2(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn3(self.fc3(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn4(self.fc4(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = self.fc5(out)
        return out

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, MetaBatchNorm1d):
                m.update_batch_stats = flag
    def update_test_stats(self, flag):
        for m in self.modules():
            if isinstance(m, MetaBatchNorm1d):
                m.test_gloabel = flag

    def reset_param(self):
        for m in self.modules():
            if isinstance(m, MetaBatchNorm1d):
                m.reset_running_stats()
            elif isinstance(m, MetaLinear):
                m.reset_parameters()


class LeNet(MetaModule):
    supported_dims = {28, 32}

    def __init__(self, state):
        if state.dropout:
            raise ValueError("LeNet doesn't support dropout")
        super(LeNet, self).__init__()
        self.conv1 = MetaConv2d(state.nc, 6, 5, padding=2 if state.input_size == 28 else 0)
        self.conv2 = MetaConv2d(6, 16, 5)
        self.fc1 = MetaLinear(16 * 5 * 5, 120)
        self.fc2 = MetaLinear(120, 84)
        self.fc3 = MetaLinear(84, 1 if state.num_classes <= 2 else state.num_classes)
    
    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


class AlexCifarNet(MetaModule):
    supported_dims = {32}
    def __init__(self, state):
        super(AlexCifarNet, self).__init__()
        assert state.nc == 3
        self.features = nn.Sequential(
            MetaConv2d(state.nc, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            MetaConv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            MetaLinear(64 * 8 * 8, 384),
            nn.ReLU(inplace=True),
            MetaLinear(384, 192),
            nn.ReLU(inplace=True),
            MetaLinear(192, state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x


# ImageNet
class AlexNet(MetaModule):
    supported_dims = {224}

    class Idt(MetaModule):
        def forward(self, x):
            return x

    def __init__(self, state):
        super(AlexNet, self).__init__()
        self.use_dropout = state.dropout
        assert state.nc == 3 or state.nc == 1, "AlexNet only supports nc = 1 or 3"
        self.features = nn.Sequential(
            MetaConv2d(state.nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MetaConv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MetaConv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MetaConv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MetaConv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if state.dropout:
            filler = nn.Dropout
        else:
            filler = AlexNet.Idt
        self.classifier = nn.Sequential(
            filler(),
            MetaLinear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            filler(),
            MetaLinear(4096, 4096),
            nn.ReLU(inplace=True),
            MetaLinear(4096, 1 if state.num_classes <= 2 else state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
