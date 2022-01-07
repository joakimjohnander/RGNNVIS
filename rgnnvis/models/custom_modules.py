
import torch
import torch.nn as nn
import torch.nn.functional as F
from rgnnvis.utils.debugging import print_tensor_statistics


"""This file contains useful custom modules that are quite modular, polished, and final."""

class LogsumexpModule(nn.Module):
    def __init__(self, cat_dim=1, accumulate_dim=(2,3)):
        super().__init__()
        self.cat_dim = cat_dim
        self.accumulate_dim = accumulate_dim
    def forward(self, x, state=None):
        """
        Args:
            x (Tensor): Of size (B, Din, M, N)
        Returns:
            Tensor: Of size (B, K * Din, M, N) where K is the number of logsumexp dimensions plus 1
        """
        x = x.where(torch.isfinite(x), torch.full_like(x, fill_value=float('-inf')))
        out = torch.cat([x]
                        + [torch.logsumexp(x, dim=dim, keepdim=True).expand_as(x)
                           for dim in self.accumulate_dim],
                        dim=self.cat_dim)
        return out

class SequentialWithState(nn.Module):
    def __init__(self, layers, return_layers=None):
        super().__init__()
        self.stateful = True
        self.return_layers = return_layers
        if len(layers) > 0 and isinstance(layers, (tuple, list)):
            for idx, layer in enumerate(layers):
                self.add_module(str(idx), layer)
        elif len(layers) > 0 and isinstance(layers, dict):
            for name, layer in layers.items():
                self.add_module(name, layer)
    def forward(self, x, state):
        if state is None:
            state = {}
        new_state = {}
        if self.return_layers is not None:
            out = {}
        for name, module in self.named_children():
            if hasattr(module, 'stateful'):
                x, new_state[name] = module(x, state.get(name))
            else:
                x = module(x)
            if self.return_layers is not None and name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        if self.return_layers is not None:
            return out, new_state
        else:
            return x, new_state

class LSTM(nn.Module):
    def __init__(self, Din, Dout, activ_naf, out_naf):
        super().__init__()
        self.stateful = True
        self.input_linear = nn.Linear(Din, 4 * Dout, bias=True)
        self.output_linear = nn.Linear(Dout, 4 * Dout, bias=True)
        self.activ_naf = activ_naf
        self.out_naf = out_naf
        self.Dout = Dout
    def get_init_state(self, B, device):
        return (torch.zeros((B, self.Dout), device=device), torch.zeros((B, self.Dout), device=device))
    def forward(self, x, state=None):
        size = x.size()
        x = x.view(-1, size[-1])
        B, D = x.size()
        if state is None:
            c, y = self.get_init_state(B, x.device)
        else:
            c, y = state
        h = self.input_linear(x) + self.output_linear(y)
        c = (self.activ_naf(h[:, 0 : self.Dout]) * torch.sigmoid(h[:, self.Dout : 2*self.Dout])
             + c * torch.sigmoid(h[:, 2*self.Dout : 3*self.Dout]))
        y = self.out_naf(c) * torch.sigmoid(h[:, 3*self.Dout : 4*self.Dout])
        state = (c, y)
        y = y.view(*size[:-1], self.Dout)
        return y, state

class LSTM2(nn.Module):
    def __init__(self, Din, Dout, activ_naf, out_naf):
        super().__init__()
        self.stateful = True
        self.input_linear = nn.Linear(Din, 4 * Dout, bias=True)
        self.output_linear = nn.Linear(Dout, 4 * Dout, bias=True)
        self.activ_naf = activ_naf
        self.out_naf = out_naf
        self.Dout = Dout
    def get_init_state(self, size, device):
        return (torch.zeros((*size, self.Dout), device=device), torch.zeros((*size, self.Dout), device=device))
    def _unpack(self, x, state):
        size = x.size()
        x = x.view(-1, size[-1])
        B, D = x.size()
        if state is None:
            state = self.get_init_state(size[:-1], x.device)
        c = state[0].view(B, self.Dout)
        y = state[1].view(B, self.Dout)
        return x, c, y, size
    def _pack(self, c, y, size):
        c = c.view(*size[:-1], self.Dout)
        y = y.view(*size[:-1], self.Dout)
        state = (c, y)
        return y, state
    def forward(self, x, state=None):
        x, c, y, size = self._unpack(x, state)

        h = self.input_linear(x) + self.output_linear(y)
        c = (self.activ_naf(h[:, 0 : self.Dout]) * torch.sigmoid(h[:, self.Dout : 2*self.Dout])
             + c * torch.sigmoid(h[:, 2*self.Dout : 3*self.Dout]))
        y = self.out_naf(c) * torch.sigmoid(h[:, 3*self.Dout : 4*self.Dout])

        return self._pack(c, y, size)

class ConvLSTM(nn.Module):
    """Note that we do not use peepholes here"""
    def __init__(self, Din, Dout, kernel_size, activ_naf, out_naf):
        super().__init__()
        self.stateful = True
        self.input_conv = nn.Conv2d(Din, 4 * Dout, kernel_size, 1, kernel_size // 2, bias=True)
        self.output_conv = nn.Conv2d(Dout, 4 * Dout, kernel_size, 1, kernel_size // 2, bias=True)
        self.activ_naf = activ_naf
        self.out_naf = out_naf
        self.Dout = Dout
    def forward(self, x, state=None):
        B, D, H, W = x.size()
        if state is None:
            c = torch.zeros((B, self.Dout, H, W), device=x.device)
            y = torch.zeros((B, self.Dout, H, W), device=x.device)
        else:
            c, y = state
        h = self.input_conv(x) + self.output_conv(y)
        c = (self.activ_naf(h[:, 0 : self.Dout]) * torch.sigmoid(h[:, self.Dout : 2*self.Dout])
             + c * torch.sigmoid(h[:, 2*self.Dout : 3*self.Dout]))
        y = self.out_naf(c) * torch.sigmoid(h[:, 3*self.Dout : 4*self.Dout])
        return y, (c, y)
        
class LinearRelu(nn.Module):
    def __init__(self, Din, Dout, naf):
        super().__init__()
        self.layer = nn.Linear(Din, Dout)
        self.naf = naf
    def forward(self, x):
        x = self.naf(self.layer(x))
        return x

class LinearChannelmod(nn.Module):
    def __init__(self, featlayers, modlayers):
        super().__init__()
        self.featlayers = featlayers
        self.modlayers = modlayers
    def forward(self, x):
        x = torch.sigmoid(self.modlayers(x)) * self.featlayers(x)
        return x

class ResidualLayer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, x):
        x = F.relu(x + self.layers(x))
        return x
class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(x + self.layers(x))
        return x
    
class IshRU(nn.Module):
    def __init__(self, Dstate, gate_layers, layers):
        super().__init__()
        self.Dstate = Dstate
        self.gate_layers = gate_layers
        self.layers = layers
    def forward(self, x):
        size = x.size()
        x = x.reshape(-1, size[-1])
        g = self.gate_layers(x)
        hnew = self.layers(x)
        hold = x[:, :self.Dstate]
        out = g[:, :self.Dstate] * hold + g[:, self.Dstate:] * hnew
        out = out.view(*size[:-1], -1)
        return out

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class GLU(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.glu(x, dim=self.dim)

class MultModule(nn.Module):
    def __init__(self, left_net, right_net, dim=-1):
        super().__init__()
        self.dim = dim
        self.left_net = left_net
        self.right_net = right_net
    def forward(self, x):
        u = self.left_net(x)
        v = self.right_net(x)
        return u * v
    
class GroupedMultModule(MultModule):
    def forward(self, x):
        u = self.left_net(x)
        v = self.right_net(x)
        return u * v.repeat_interleave(repeats=(u.size(self.dim) // v.size(self.dim)), dim=self.dim)

class SequentialResidual(nn.Sequential):
    def __init__(self, dim_min, dim_max, *parent_args, **parent_kwargs):
        super().__init__(*parent_args, **parent_kwargs)
        self.dim_min = dim_min
        self.dim_max = dim_max
    def forward(self, x):
        size = x.size()
        x = x.view(-1, size[-1])
        h = x
        for layer in self:
            h = layer(h)
        h = x[:, self.dim_min : self.dim_max] + h
        h = h.view(*size[:-1], self.dim_max - self.dim_min)
        return h

class BNWrap(nn.Module):
    def __init__(self, dim, *parent_args, **parent_kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*parent_args, **parent_kwargs)
        self.dim = dim
    def forward(self, x):
        assert self.dim + 1 == x.dim()
        original_size = x.size()
        x = x.view(-1, original_size[-1])
        x = self.bn(x)
        x = x.view(*original_size)
        return x

class ParallellSum(nn.ModuleList):
    def forward(self, *args, **kwargs):
        return sum([layer(*args, **kwargs) for layer in self])
class ParallelCat(nn.ModuleList):
    def forward(self, *args, **kwargs):
        return torch.cat([layer(*args, **kwargs) for layer in self], dim=-1)

class Add(nn.ModuleList):
    def __init__(self, in_layers, out_layer):
        super().__init__(in_layers)
        self.out_layer = out_layer
    def forward(self, *args):
        return self.out_layer(sum([self[idx](arg) for idx, arg in enumerate(args)]))
class Concat(nn.ModuleList):
    def __init__(self, in_layers, out_layer):
        super().__init__(in_layers)
        self.out_layer = out_layer
    def forward(self, *args):
        return self.out_layer(torch.cat([self[idx](arg) for idx, arg in enumerate(args)], -1))

class Difference(nn.ModuleDict):
    def forward(self, x, y):
        return self['left_in'](x) - self['right_in'](y)
class Hadamard(nn.ModuleDict):
    def forward(self, x, y):
        return self['left_in'](x) * self['right_in'](y)
class Cat(nn.ModuleDict):
    def forward(self, x, y):
        return torch.cat([self['left_in'](x), self['right_in'](y)], dim=-1)
    
class Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self, x):
        return torch.tensor(self.value, device=x.device)
