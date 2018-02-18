import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm   # Progress bar
import numpy as np
#from torchvision.utils import save_image

class VAE_Net(nn.Module):
    
    # MAIN VAE Class

    def __init__(self):  ### TODO: Change initflag to be in input? 
        super(VAE_Net, self).__init__()

        # define the encoder and decoder
        self.dataset = 'Frey'
        self.latent = 20
        self.iniflag = True # Controls weights initalisation. True draws weights from N(0,0.01). Should this be inside
        # VAE class or implenented in an utilities module as de weight_init(m) in a sep util module
        if self.dataset is 'Frey':
            self.layesize = 200
        else:
            self.layesize = 500
        self.stddev = 0.01
        self.ei = nn.Linear(28 * 28 * 1, self.layesize)
        self.em = nn.Linear(self.layesize, self.latent)
        self.ev = nn.Linear(self.layesize, self.latent)

        self.di = nn.Linear(self.latent, self.layesize)
        self.dom = nn.Linear(self.layesize, 28 * 28 * 1)
        self.dov = nn.Linear(self.layesize, 28 * 28 * 1)
        
        if self.initFlag is True:
            for m in self.modules():
                m.weight.data.normal_(0,self.stddev)
            
    
    def encode(self, x):

        # encoder part

        o = F.tanh(self.ei(x))
        mu = self.em(o)
        logvar = self.ev(o) 
        print("Encoder output Mean Size:"+ " "+str(mu.size())+"\n")
        print("Encoder output Variance Size:"+str(logvar.size())+"\n")
        return mu, logvar

    def decode(self, x):

        # decoder part    

        o = F.tanh(self.di(x))
        im = F.sigmoid(self.dom(o))
        ivar = self.dov(o)
        print("Decoder output Mean Size:"+ " "+str(im.size())+"\n")
        print("Encoder output Variance Size:"+str(ivar.size())+"\n")
        return im,ivar

    def sample(self):

        # get a N(0,1) sample in a torch/cuda tensor        

        return Variable(torch.randn(self.latent).cuda(), requires_grad = False)

    def repar(self, mu, logvar):

        # the infamous reparamaterization trick (aka 4 lines of code)

        samp = self.sample()
        samp = F.mul(torch.rsqrt(logvar.exp()),samp)
        samp = samp + mu
        return samp

    def forward(self, x):

        # forward pass (take your image, get its params, reparamaterize the N(0,1) with them, decode and output)

        mu, logvar = self.encode(x)
        om, ov = self.decode(self.repar(mu,logvar))
        return om,ov , mu, logvar



def elbo_loss(mu, logvar, x, x_pr):

    # ELBO loss; NB: the L2 Part is not necessarily correct
    # BCE actually seems to work better, which tries to minimise informtion loss (in bits) between the original and reconstruction
    print("ELBO mu dims are:"+ " "+str(mu.size())+"\n")
    print("ELBO logvar dims are:"+ " "+str(logvar.size())+"\n")
    print("Input dims are:"+" "+str(x.size())+"\n")
    size = mu.size() # get size of the distribution
    # Basic error handling 
    if logvar.exp().eq(0).any():
        raise Exception('fml zero variance')
    # Calculate reconstruction error 
    denum = torch.prod((2*np.pi*logvar.exp())) # determinant is the product of diag elems
    coeff = -denum.sqrt().log() # take the log
    cov_inv = 1. /logvar.exp() # compute the inverse covariance matrix 
    exponent = -0.5*torch.prod((x - mu)**2,torch.t(cov_inv)) # compute the exponent
    Recon_part = coeff - exponent; # the reconstruction part is -log(p(x|z)) so the signs are reversed
    ### Calculate KL divergence 
    KL_part = 0.5*((logvar.exp().sum() + mu.dot(mu) - size[0]*size[1] - logvar.sum()))
    ### Calculate regulariser
    params = torch.cat(mu,logvar) # We need to regularise the decoder parameters using a prior N(0,I) over \theta
    params_size = params.size() # Require for calculating scaling constant
    Regulariser = -(-0.5*((params**2).sum())-params_size[0]*params_size[1]*(2*np.pi).log()) # -ve sign because we minimise the NLL so we need -log(p(theta))
    # Recon_part = F.binary_cross_entropy(x_pr, x, size_average=False)
    #Recon_part = F.mse_loss(x_pr, x, size_average=False)
    #print('L2 loss: %.6f' % L2_part)
    #print('kL loss: %.6f' % KL_part)
    return Recon_part + KL_part + Regulariser



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



def train(model, optimizer, train_loader, loss_func, epochs = 1, show_prog = 100):

    # stolen from a generic pytorch training implementation
    # TODO: Train on different data
    
    model.train()
    for i in tqdm(range(epochs)):
        for batch_idx, (data, _ ) in enumerate(train_loader):
            
            data = Variable(data, requires_grad = False).view(-1,784)  # NEED TO FLATTEN THE IMAGE FILE
            data = data.cuda()  # Make it GPU friendly
            optimizer.zero_grad()   # reset the optimzer so we don't have grad data from the previous batch
            output, mu, var = model(data)   # forward pass
            loss = loss_func(mu, var, data, output) # get the loss
            loss.backward() # back prop the loss
            optimizer.step()    # increment the optimizer based on the loss (a.k.a update params?)
            #print('Batch Training Loss is: %.6f' % loss[0])
            if batch_idx % show_prog == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
