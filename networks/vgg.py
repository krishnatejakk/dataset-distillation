'''VGG11/13/16/19 in Pytorch.'''


import torch
import torch.nn as nn
# from . import utils


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    supported_dims = {28, 32}
    def __init__(self, vgg_name, in_channels=3, input_size=32):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.features = self._make_layers(cfg[vgg_name])
        self.embDim = 512
        self.classifier = nn.Linear(512, 10)
       
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if (self.input_size == 28) and (i == 0) else 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.features(x)
                e = out.view(out.size(0), -1)
        else:
            out = self.features(x)
            e = out.view(out.size(0), -1)
        out = self.classifier(e)
        if last:
            return out, e
        else:
            return out
        
    def get_embedding_dim(self):
        return self.embDim

class VGG11(nn.Module):
    supported_dims = {28, 32}
    def __init__(self, state):
        super(VGG11, self).__init__()
        self.in_channels = state.nc
        self.features = self._make_layers(cfg['VGG11'])
        self.embDim = 512
        self.classifier = nn.Linear(512, state.num_classes)
        self.state = state
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if (self.state.input_size == 28) and (i == 0) else 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.features(x)
                e = out.view(out.size(0), -1)
        else:
            out = self.features(x)
            e = out.view(out.size(0), -1)
        out = self.classifier(e)
        if last:
            return out, e
        else:
            return out

def test():
    net = VGG('VGG11', in_channels=1, input_size=28)
    x = torch.randn(2,1,28,28)
    y = net(x)
    print(y.size())

if __name__ == "__main__":
    test()
# test()
