import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class VAE_Net(nn.Module):

    def __init__(self):
        super(VAE_Net, self).__init__()

        # define the encoder and decoder

        self.latent = 20

        self.ei = nn.Linear(28 * 28 * 1, 500)
        self.em = nn.Linear(500, self.latent)
        self.ev = nn.Linear(500, self.latent)

        self.di = nn.Linear(self.latent, 500)
        self.do = nn.Linear(500, 28 * 28 * 1)

    def encode(self, x):
        o = F.sigmoid(self.ei(x))
        mu = self.em(o)
        logvar = self.ev(o)
        return mu, logvar

    def decode(self, x):
        o = F.sigmoid(self.di(x))
        im = F.sigmoid(self.do(o))
        return im

    def sample(self):
        return Variable(torch.randn(self.latent).cuda())

    def repar(self, mu, logvar):
        samp = self.sample()
        samp = F.mul((0.5*logvar).exp(),samp)
        samp = samp + mu
        return samp

    def forward(self, x):
        mu, logvar = self.encode(x)
        f = self.decode(self.repar(mu,logvar))
        return f, mu, logvar

def elbo_loss(mu, logvar, x, x_pr):
    size = mu.size()
    KL_part = 0.5*((logvar.exp().sum() + mu.dot(mu) - size[0]*size[1] - logvar.sum()))
    L2_part = F.binary_cross_entropy(x_pr, x, size_average=False)
    #print('L2 loss: %.6f' % L2_part)
    #print('kL loss: %.6f' % KL_part)
    return L2_part + KL_part

def get_data_loaders(b_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
                        batch_size=b_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.ToTensor()),
                        batch_size=b_size, shuffle=True, **kwargs)
    return train_loader, test_loader

def train(model, optimizer, train_loader, loss_func, epochs = 1):
    model.train()
    for i in range(epochs):
        for batch_idx, (data, _ ) in enumerate(train_loader):
            data = Variable(data).view(-1,784)
            data = data.cuda()
            optimizer.zero_grad()
            output, mu, var = model(data)
            loss = loss_func(mu, var, data, output)
            loss.backward()
            optimizer.step()
            #print('Batch Training Loss is: %.6f' % loss[0])
