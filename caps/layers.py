import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def squash(s, dim=-1):
    '''Perform the operation:

    n^2 / (n^2 + 1) * s / n,

    where n = L2 norm of a vector.

    Arguments:

    `s`: torch.Tensor.

    `dim`: Calculate norm on the 'dim'-th dimension.

    Returns:

    Scaled `s`.'''

    # sum over features of each capsule in each batch
    scale = (s ** 2).sum(dim=dim, keepdim=True) ** 0.5
    s = s / scale
    s[s == np.nan] = 0
    scale = scale ** 2
    scale = scale / (1 + scale)
    return scale * s


class PrimaryCaps(nn.Module):
    '''Primary Capsule layer, in which capsules are convolutional.'''
    def __init__(self, n_capsules, in_channels, capsule_dimension,
                kernel:int, stride:int):
        '''Arguments:

        `n_capsules`: number of the capsules in the layer.

        `in_channels`: number of channels of input data.

        `capsule_dimension`: dimension of convolutional capsules,
        essentially number of channels of the resulting volume.

        `kernel`: (int) dimension of square kernel.

        `stride`: (int) stride of kernel.'''
        super().__init__()

        # represent conv capsules as 1 convolution with capsule_dimension*n_capsules
        # channels which is to be split later
        self.capsules = nn.Conv2d(in_channels, capsule_dimension*n_capsules,
                                kernel_size=kernel, stride=stride)
        self.n_capsules = n_capsules

    def forward(self, x):
        # x expected to be (batch, in_channels, h, w)
        # (batch, n_capsules, features)
        s = self.capsules(x).view(x.size(0), self.n_capsules, -1)
        return self.squash(s)

    @staticmethod
    def squash(s):
        return squash(s)

class SemanticCaps(nn.Module):
    '''Second+ layer of a CapsNet-like architecture.'''
    def __init__(self, n_capsules, n_prev_capsules,
                capsule_dimension, input_size,
                routing_iterations, gpu=True):
        '''Arguments:

        `n_capsules`: number of the capsules in the layer.

        `n_prev_capsules`: number of the capsules in the previous layer.

        `capsule_dimension`: dimension of capsules in the layer.

        `input_size`: 'dimension' of the input capsules.

        `routing_iterations`: iterations of dynamic routing.

        `gpu`: bool, if environment supports a GPU.'''
        super().__init__()
        # small weights circumvent saturation issues in dynamic routing
        self.Ws = nn.Parameter(1e-3 * torch.randn(n_capsules, n_prev_capsules,
                                          capsule_dimension, input_size))
        self.route_iters = routing_iterations
        self.gpu = gpu

    def forward(self, x):
        # expected to be (batch, caps_in, f_in)

        # add 'batch' dimension, now (1, caps_out, caps_in, f_out, f_in)
        Ws = self.Ws[None, ...]
        # add 'caps_out' and dummy dim at the end for matmul to work, now
        # (batch, 1, caps_in, f_in, 1)
        x = x[:, None, ..., None]

        if self.gpu:
            Ws = Ws.cuda()

        # u_hat dims:
        # (1, caps_out, caps_in, f_out, f_in) <matmul> (batch, 1, caps_in, f_in, 1) =
        # (batch, caps_out, caps_in, f_out, 1), + squeeze =>
        # (batch, caps_out, caps_in, f_out)
        u_hat = Ws.matmul(x).squeeze(dim=-1) # get rid of dummy dim at the end

        # b -> (batch, caps_out, caps_in)
        v = self.routing(u_hat, b=torch.zeros(*u_hat.size()[:-1]))

        return v

    @staticmethod
    def squash(s):
        return squash(s)

    def routing(self, u_hat, b):
        # b: (batch, caps_out, caps_in)
        if self.cpu:
            b = b.cuda()

        for i in range(self.route_iters):
            # softmax on caps_out => for every input capsule
            # same dim as b
            c = F.softmax(b, dim=1)
            # expand at dim 2 for matmul to work, now
            # (batch, caps_out, 1, caps_in)
            c = c[:, :, None, :]

            if self.cpu:
                c = c.cuda()

            # s dims:
            # (batch, caps_out, 1, caps_in) <matmul> (batch, caps_out, caps_in, f_out) =
            # (batch, caps_out, 1, f_out), + squeeze =>
            # (batch, caps_out, f_out)
            s = c.matmul(u_hat).squeeze(dim=2)

            v = self.squash(s)

            if i == self.route_iters - 1: break  # no need to update logits if last iteration
                                                # could return here but keep classy

            # insert dummy dim for matmul to work, now
            # (batch, caps_out, f_out, 1)
            v = v[..., None]

            # matmul dims:
            # (batch, caps_out, caps_in, f_out) <matmul> (batch, caps_out, f_out, 1) =
            # (batch, caps_out, caps_in, 1), + squeeze =>
            # (batch, caps_out, caps_in)
            b = b + u_hat.matmul(v).squeeze(dim=-1)

        return v

# NOTE: try transpose convs
# NOTE: try iterative 

class FFDecoder(nn.Module):
    '''Simple Feedforward Net, returns batched images
    as specified by input arguments'''

    def __init__(self, layers, output_size, input_size,
                act_fun=torch.sigmoid, gpu=True):
        '''Arguments:
        
        `layers`: iterable containing number of hidden units per layer.

        `output_size`: 3D tuple containing channels, height and width.

        `input_size`: tuple containing number of capsules
        and their dimension.

        `act_fun`: activation function at the output layer.
        Default: `torch.sigmoid`.

        `gpu`: bool, if environment supports a GPU.'''
        super().__init__()

        self.output_size = output_size
        self.act_fun = act_fun
        self.gpu = gpu

        try:
            input_flattened = input_size[0] * input_size[1]
        except TypeError:
            # if single capsule per network is used for reconstruction
            input_flattened = input_size

        # channels * height * width
        output_flattened = output_size[0] * output_size[1] * output_size[2]
        self.layers = nn.ModuleList([
        # feedforward net
            nn.Linear(in_nodes, out_nodes) for out_nodes, in_nodes \
            in zip(layers + [output_flattened], [input_flattened] + layers)
        ])

    def forward(self, x):
        # x expected to be (batch, capsules, capsule_dimension)

        # flatten capsules
        x = x.view(x.size(0), -1)

        for layer, f in zip(self.layers,
                           (len(self.layers) - 1) * [F.relu] + [self.act_fun]):
            if self.gpu:
                layer = layer.cuda()
            x = f(layer(x))

        # reshape to specified width, height and channels, keep batched
        x = x.reshape(x.size(0), *self.output_size)
        return x
