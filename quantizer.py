import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.ticker import PercentFormatter

'''
out = []
out = torch.zeros(4).cuda()
out = 0
total = 0
'''
def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantizeCSQ(v, s, p, isActivation=False):
    global out
    if isActivation:
        Qn = 0
        Qp = 2**p - 1
        #print('sym_a')
        #v = torch.erf(v / 1.4142135624)
        gradScaleFactor = 1.0 / math.sqrt(v.shape[1] * (2**p))
        #gradScaleFactor = 1.0 / math.sqrt(v.numel() * (2**p))
        s = grad_scale(s, gradScaleFactor)
        vbar = round_pass((v / s).clamp(Qn, Qp))
        #print(vbar)

        vhat = vbar * s

        #hist = torch.histc(vbar,bins=4)
        #print(hist)

        return vhat
    else: # is wei
        # ght
        Qn = -2**(p-1) + 0.5
        Qp = 2**(p-1) - 0.5
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * (2**p))
        #print('sym-W')
        #quantize
        s = grad_scale(s, gradScaleFactor)
        vbar = ((round_pass(((v/s)-0.5)))+0.5).clamp(Qn, Qp)
        vhat = vbar * s
        #print(vbar)
        #print(torch.histc(vbar))


        return vhat
def quantizeCSQ_affine(v, s, z, p):
    #global out
    # is wei
    # ght
    Qn = -2**(p-1) + 0.5
    Qp = 2**(p-1) - 0.5
    gradScaleFactor = 1.0 / math.sqrt(v.numel() * (2**p))
    #print('sym-W')
    #quantize
    s = grad_scale(s, gradScaleFactor)
    z = grad_scale(z, gradScaleFactor)
    vbar = ((round_pass(((v/s+z)-0.5)))+0.5).clamp(Qn, Qp)-z
    vhat = vbar * s
    #print(z)
    #print(vbar)
    #print(torch.histc(vbar))


    return vhat
def quantizeCSQ_EWGS(v, u, l, s, p, isActivation=False):
    if isActivation:
        Qn = 0
        Qp = 2**p - 1
        #print('sym_a')
        #v = torch.erf(v / 1.4142135624)
        gradScaleFactor = 1.0 / math.sqrt(v.shape[1] * (2**p))
        #gradScaleFactor = 1.0 / math.sqrt(v.numel() * (2**p))
        s = grad_scale(s, gradScaleFactor)
        vbar = round_pass((v / s).clamp(Qn, Qp))
        #print(vbar)

        vhat = vbar * s

        #hist = torch.histc(vbar,bins=4)
        #print(hist)


        return vhat
    else: # is wei
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * (2**p))
        s = grad_scale(s, gradScaleFactor)
        u = grad_scale(u, gradScaleFactor)
        l = grad_scale(l, gradScaleFactor)
        vbar = round_pass(((v - l) / (u - l)).clamp(min=0, max=1) * (2 ** p - 1)) / (2 ** p - 1)
        vhat = (vbar - 0.5) * 2 # [-1, 1]
        vbar = vbar * s
        return vhat

def quantizeLSQ(v, s, p, numl, isActivation=False):
    global out, total
    if isActivation:
        Qn = 0
        Qp = 2**p -1
        #print("LSQ-A")
        #gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)
        gradScaleFactor = 1.0 / math.sqrt(numl* (Qp))
        s = grad_scale(s, gradScaleFactor)
        vbar = round_pass((v / s).clamp(Qn, Qp))
        #print(torch.histc(vbar))
        vhat = vbar * s
        #print(vbar)

    else: # is weight
        Qn = -2**(p-1)
        Qp = 2**(p-1) - 1
        gradScaleFactor = 1.0 / math.sqrt(numl * (Qp))
        #print("LSQ-W")
        #quantize

        s = grad_scale(s, gradScaleFactor)
        vbar = round_pass((v / s).clamp(Qn, Qp))
        #vbar = ((round_pass(((v/s)+0.5)))-0.5).clamp(Qn, Qp)
        # print(torch.histc(vbar))

        vhat = vbar * s



        #out = out + torch.histc(vbar, bins=4)  # .flatten().detach().cpu().numpy())
        #out = out + list((v/s).flatten().detach().cpu().numpy())
        #dist = (out * 100 / out.sum())
        #print()

        #plt.bar(np.arange(-2.5,1.5,1), dist, width=0.5)
        #plt.hist(out,bins=100, density=True)


        #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))


        #plt.draw()
        #plt.pause(0.01)
        #plt.savefig("/home/faaiz/Documents/2021/trained/lsq")
        #plt.clf()





        '''
        out = out + list(((round_pass(((v/(0.68*torch.std(v)))+0.5)))-0.5).clamp(Qn, Qp).flatten().detach().cpu().numpy())

        ###         histogram           ###
        hist = np.histogram(out, bins=[-2, -1, 0, 1, 2])

        hist = hist[0] / (hist[0].sum())
        print(hist)
        vbar.cuda()

        ###         GRAPH           ###
        # vbar.cuda()
        plt.hist(out)
        plt.draw()
        plt.pause(0.01)
        plt.clf()
        '''

    return vhat



class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.nbits = kwargs['nbits']
        self.step_size_w = Parameter(torch.Tensor(1))
        #self.step_size_w = Parameter(torch.Tensor(out_channels))
        self.step_size_a = Parameter(torch.Tensor(1))
        self.z =  Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))



class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.nbits = kwargs['nbits']
        #self.step_size_w = Parameter(torch.Tensor(1))
        self.step_size_w = Parameter(torch.Tensor(1))
        self.step_size_a = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

class _ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.nbits = kwargs_q['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))


class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=8):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits)

    def forward(self, x):

        if self.init_state == 0:
            #print('initialized')
            self.step_size_w.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            #self.step_size_w.data.copy_(2 * self.weight.abs().view(self.weight.size(0), -1).mean(-1) / math.sqrt(2 ** (self.nbits - 1) - 1))
            if (x.shape[1] == 3):
                self.step_size_a.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            else:
                self.step_size_a.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            #self.step_size_a.data.copy_(2 * x.abs().view(x.shape[1],x.shape[0]*x.shape[2]*x.shape[3]).mean(dim = -1) / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        #print(self.step_size_a.shape)
        if (x.shape[1] == 3):
            x_q = quantizeLSQ(x, self.step_size_a, self.nbits,x.shape[1])
        else:
            x_q = quantizeLSQ(x, self.step_size_a, self.nbits, x.shape[1], isActivation=True)
        w_q = quantizeLSQ(self.weight, self.step_size_w, self.nbits, self.weight.data.numel())

        return F.conv2d(x_q, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv2dCSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=2):
        super(Conv2dCSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size_w.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.step_size_a.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            #self.u.data.fill_(self.weight.std()*3.0)
            #self.l.data.fill_(-self.weight.std()*3.0)
            self.z.data.copy_(self.weight.mean())
            self.init_state.fill_(1)

        #print(self.nbits)
        #global out
        x_q = quantizeCSQ(x, self.step_size_a, self.nbits, isActivation=True)

        #out = out + torch.histc(x_q,bins=4)
        #print(out*100/out.sum())
        #print(x_q/self.step_size_a)
        w_q = quantizeCSQ_affine(self.weight, self.step_size_w, self.z, self.nbits)
        #w_q = quantizeCSQ_EWGS(self.weight, self.u, self.l, self.step_size_w, self.nbits)
        #out = out + torch.histc(w_q,bins=4)
        #print(out*100/out.sum())
        #print(w_q / self.step_size_w)
        output =  F.conv2d(x_q, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output



class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=8):
        super(LinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size_w.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.step_size_a.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        w_q = quantizeLSQ(self.weight, self.step_size_w, self.nbits, self.weight.data.numel() )
        x_q = quantizeLSQ(x, self.step_size_a, self.nbits, x.shape[1], isActivation=True)

        return F.linear(x_q, w_q, self.bias)

class LinearLSQ_sym(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=8):
        super(LinearLSQ_sym, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size_w.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.step_size_a.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)



        w_q = quantizeLSQ_sym(self.weight, self.step_size_w, self.nbits,)
        x_q = quantizeLSQ_sym(x, self.step_size_a, self.nbits, isActivation=True)

        return F.linear(x_q, w_q, self.bias)

class ActLSQ(_ActQ):
    def __init__(self, nbits=8):
        super(ActLSQ, self).__init__(nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        x_q = quantizeLSQ(x, self.step_size, self.nbits, isActivation=True)

        return x_q

class ActLSQ_sym(_ActQ):
    def __init__(self, nbits=8):
        super(ActLSQ_sym, self).__init__(nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        x_q = quantizeLSQ_sym(x, self.step_size, self.nbits, isActivation=True)

        return x_q
