import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
#from torchvision.utils import save_image

class VAE_Net(nn.Module):
    
    # MAIN VAE Class

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

        # encoder part

        o = F.sigmoid(self.ei(x))
        mu = self.em(o)
        logvar = self.ev(o)
        return mu, logvar

    def decode(self, x):

        # decoder part    

        o = F.sigmoid(self.di(x))
        im = F.sigmoid(self.do(o))
        return im

    def sample(self):

        # get a N(0,1) sample in a torch/cuda tensor        

        return Variable(torch.randn(self.latent).cuda())

    def repar(self, mu, logvar):

        # the infamous reparamaterization trick (aka 4 lines of code)

        samp = self.sample()
        samp = F.mul((0.5*logvar).exp(),samp)
        samp = samp + mu
        return samp

    def forward(self, x):

        # forward pass (take your image, get its params, reparamaterize the N(0,1) with them, decode and output)

        mu, logvar = self.encode(x)
        f = self.decode(self.repar(mu,logvar))
        return f, mu, logvar



def elbo_loss(mu, logvar, x, x_pr):

    # ELBO loss; NB: the L2 Part is not necessarily correct, BCE actually seems to work better for some reason?
    # TODO: make the reconstruction error resemble the papers

    size = mu.size()
    KL_part = 0.5*((logvar.exp().sum() + mu.dot(mu) - size[0]*size[1] - logvar.sum()))
    L2_part = F.mse_loss(x_pr, x, size_average=False)
    #print('L2 loss: %.6f' % L2_part)
    #print('kL loss: %.6f' % KL_part)
    return L2_part + KL_part



def get_data_loaders(b_size):

    # downloads the MNIST data, outputs these PyTorch wrapped data loaders
    # TODO: MAKE THIS DATASET AGNOSTIC

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

    # stolen from a generic pytorch training implementation
    # TODO: Train on different data

    model.train()
    for i in range(epochs):
        for batch_idx, (data, _ ) in enumerate(train_loader):
            data = Variable(data).view(-1,784)  # NEED TO FLATTEN THE IMAGE FILE
            data = data.cuda()  # Make it GPU friendly
            optimizer.zero_grad()   # reset the optimzer so we don't have grad data from the previous batch
            output, mu, var = model(data)   # forward pass
            loss = loss_func(mu, var, data, output) # get the loss
            loss.backward() # back prop the loss
            optimizer.step()    # increment the optimizer based on the loss (a.k.a update params?)
            #print('Batch Training Loss is: %.6f' % loss[0])
