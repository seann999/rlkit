"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parallel import parallel_apply, gather
from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm

import time

def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=torch.nn.init.xavier_uniform_,#ptu.fanin_init,
            b_init_value=0,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(0)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        hidden_init(self.last_fc.weight)
        self.last_fc.bias.data.fill_(b_init_value)
        #self.last_fc.weight.data.uniform_(-init_w, init_w)
        #self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
        
class SplitMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            heads,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=torch.nn.init.xavier_uniform_,#ptu.fanin_init,
            b_init_value=0,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.heads = heads
        self.convs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Conv1d(in_size * heads, next_size * heads, 1, groups=heads)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(0)
            self.__setattr__("fc{}".format(i), fc)
            self.convs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_conv = nn.Conv1d(in_size * heads, output_size * heads, 1, groups=heads)
        hidden_init(self.last_conv.weight)
        self.last_conv.bias.data.fill_(b_init_value)
        #self.last_fc.weight.data.uniform_(-init_w, init_w)
        #self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        if len(input[0]) == self.heads:
            tensors = [vec for vecs in input[0] for vec in vecs]
            h = torch.cat(tensors, 1).unsqueeze(2)
        else:
            h = input.unsqueeze(2).repeat(1, self.heads, 1)
            
        for i, conv in enumerate(self.convs):
            h = conv(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_conv(h)

        output = self.output_activation(preactivation)
        output = output.squeeze(2)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class EnsembleFlattenMlp(PyTorchModule):
    def __init__(self, n_nets, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.nets = []

        for i in range(n_nets):
            mlp = Mlp(**kwargs)
            self.nets.append(mlp)
            self.add_module('mlp_' + str(i), mlp)

    def forward(self, *inputs, **kwargs):
        start = time.time()
        
        if len(inputs[0]) == len(self.nets):
            outputs = parallel_apply(self.nets, [torch.cat(tup, dim=1) for tup in inputs[0]], devices=[i // 4 for i in range(len(self.nets))])
        else:
            outputs = parallel_apply(self.nets, [torch.cat(inputs, dim=1) for _ in range(len(self.nets))], devices=[i // 4 for i in range(len(self.nets))])
        #outputs = []

        #for net in self.nets:
        #out = net.forward(flat_inputs, **kwargs)
        #outputs.append(out)

        #flat_outputs = torch.cat(outputs, dim=1)
        print(time.time() - start)

        flat_outputs = gather(outputs, 0, 1)

        return flat_outputs
    
class SplitFlattenMlp(SplitMlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        #print(len(inputs), len(inputs[0]), isinstance(inputs, list), isinstance(inputs[0], list))
        if isinstance(inputs[0], list):
            flat_inputs = inputs
        else:
            flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)
