import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm   # Progress bar
import numpy as np

import os
from urllib.request import urlopen
from scipy.io import loadmat

from tensorboardX import SummaryWriter

#from torchvision.utils import save_image


class VAE_Net(nn.Module):
    
    # MAIN VAE Class

    def __init__(self, latent_size=20, data='MNIST'):
        super(VAE_Net, self).__init__()

        # define the encoder and decoder
        
        self.data = data

        if self.data =='MNIST':
            self.h = 28
            self.w = 28
            self.u = 500
        elif self.data == 'Frey':
            self.h = 28
            self.w = 20
            self.u = 200
            # add variance layer for Gaussian output
            self.dov = nn.Linear(self.u, self.h * self.w * 1)
        elif self.data == 'cifar':
            self.h = 32
            self.w = 32
            self.u = 200

        self.latent = latent_size

        self.cl1 = nn.Conv2d(1, 30, 8)
        self.p1 = nn.MaxPool2d(3)
        self.cl2 = nn.Conv2d(30, 80, 6)

        self.ei = nn.Linear(320, self.u)
        self.em = nn.Linear(self.u, self.latent)
        self.ev = nn.Linear(self.u, self.latent)

        self.di = nn.Linear(self.latent, self.u)
        self.dom = nn.Linear(self.u, 320)#(self.u, self.h * self.w * 1)
        self.dcl2 = nn.ConvTranspose2d(80, 30, 6)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=3)
        self.dcl1 = nn.ConvTranspose2d(30, 1, 8)

        """
        self.cl1 = nn.Conv2d(1, 64, 7)#, stride=2)
        self.p1 = nn.MaxPool2d(2)
        self.cl2 = nn.Conv2d(64, 64, 3)
        self.cl3 = nn.Conv2d(64, 128, 3)#, stride=2)
        self.cl4 = nn.Conv2d(128, 256, 3)#, stride=2)
        self.cl5 = nn.Conv2d(256, 512, 3)#, stride=2)
        #self.avg_pool = nn.AvgPool2d(3)

        self.ei = nn.Linear(512, self.u)#(self.h * self.w * 1, self.u)
        self.em = nn.Linear(self.u, self.latent)
        self.ev = nn.Linear(self.u, self.latent)

        self.di = nn.Linear(self.latent, self.u)
        self.dom = nn.Linear(self.u, 512)#(self.u, self.h * self.w * 1)

        self.avg_unpool = nn.ConvTranspose2d(512, 512, 3)
        self.dcl5 = nn.ConvTranspose2d(512, 256, 3)
        self.dcl4 = nn.ConvTranspose2d(256, 128, 3)
        self.dcl3 = nn.ConvTranspose2d(128, 64, 3)
        self.dcl2 = nn.ConvTranspose2d(64, 64, 3)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)#nn.MaxUnpool2d(2)
        self.dcl1 = nn.ConvTranspose2d(64, 1, 7)
        self.softmax = nn.Softmax2d()
        """


    def encode(self, x):

        # encoder part
        x = F.relu(self.cl1(x))
        x = F.relu(self.p1(x))
        x = F.relu(self.cl2(x))
        #x = self.cl3(x)
        #x = self.cl4(x)
        #x = self.cl5(x)
        #self.size_before_ap = x.size()[-1]
        #x = nn.AvgPool2d(self.size_before_ap)(x)
        x = x.view(x.size()[0], -1)#x.size()[1])

        o = F.tanh(self.ei(x))
        mu = self.em(o)
        logvar = self.ev(o)
        return mu, logvar

    def decode(self, x):

        # decoder part    
        
        # THIS IS THE ORIGINAL
        #o = F.sigmoid(self.di(x))
        #im = F.sigmoid(self.do(o))
        #return im

        o = F.tanh(self.di(x))
        im = F.sigmoid(self.dom(o))
        im = im.view(im.shape[0], 80, -1)
        im = im.view(im.shape[0], 80, int(im.shape[2]/2), int(im.shape[2]/2))

        #im = im.unsqueeze(-1).unsqueeze(-1)
        #im = self.avg_unpool(im)
        #im = self.dcl5(im)
        #im = self.dcl4(im)
        #im = self.dcl3(im)
        im = F.relu(self.dcl2(im))
        im = F.relu(self.up1(im))
        im = F.sigmoid(self.dcl1(im))
        #im = self.softmax(im)

        if self.data == 'Frey':
            ivar = self.dov(o)
        # print("Decoder output Mean Size:"+ " "+str(im.size())+"\n")
        # print("Encoder output Variance Size:"+str(ivar.size())+"\n")
            return im,ivar
        else:
            return im, []

    def sample(self):

        # get a N(0,1) sample in a torch/cuda tensor        

        return Variable(torch.randn(self.latent).cuda(), requires_grad = False)
        #return Variable(torch.randn(self.latent), requires_grad=False)

    def repar(self, mu, logvar):

        # the infamous reparamaterization trick (aka 4 lines of code)

        samp = self.sample()
        samp = F.mul((0.5*logvar).exp(),samp)
        samp = samp + mu
        return samp

    def forward(self, x):

        # forward pass (take your image, get its params, reparamaterize the N(0,1) with them, decode and output)

        #mu, logvar = self.encode(x)
        #f = self.decode(self.repar(mu,logvar))
        #return f, mu, logvar

        mu, logvar = self.encode(x)
        om, ov = self.decode(self.repar(mu,logvar))
        return om, ov , mu, logvar

def elbo_loss(enc_m, enc_v, x, dec_m, dec_v, model):

    # ELBO loss; NB: the L2 Part is not necessarily correct
    # BCE actually seems to work better, which tries to minimise informtion loss (in bits) between the original and reconstruction
    # TODO: make the reconstruction error resemble the papers

    dec_m = dec_m.view(dec_m.size()[0], -1)
    x = x.view(x.size()[0], -1)

    size = enc_m.size()

    KL_part = 0.5*((enc_v.exp().sum() + enc_m.dot(enc_m) - size[0]*size[1] - enc_v.sum()))
    
    if model.data == 'Frey':    # get the complicated Gaussian reconstruction term

        Recon_part = torch.sum(    torch.sum(    ((x - dec_m)**2)*(1./dec_v.exp()),dim=1    )   )

        Pi_term = Variable((0.5 * torch.Tensor([2*np.pi]).log() * size[0]).cuda())
        #Pi_term = Variable((0.5 * torch.Tensor([2 * np.pi]).log() * size[0]))
        
        Recon_norm = torch.sum(0.5 * torch.sum(dec_v, dim = 1))
        
        Recon_total = Recon_part + Recon_norm + Pi_term
    
    else:   # assume MNIST, therefore 'pseudo-binary'
        
        Recon_total = F.binary_cross_entropy(dec_m, x, size_average=False)
    
    #print('Recon loss: %.6f' % (Recon_part))
    #print('KL loss: %.6f' % KL_part)
    #print('MSE loss: %.6f' % MSE_part)
    #print('Variance sum: %.6f' % dec_v.sum())

    output = Recon_total + KL_part

    return(output)


def get_data_loaders(b_size, data = 'MNIST'):

    # downloads the MNIST data, outputs these PyTorch wrapped data loaders
    # TODO: MAKE THIS DATASET AGNOSTIC

    kwargs = {'num_workers': 1, 'pin_memory': True}
    
    if data == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.ToTensor()),
                            batch_size=b_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                           transform=transforms.ToTensor()),
                            batch_size=b_size, shuffle=True, **kwargs)
    elif data == 'Frey':
        check_frey()
        # reshape data for later convenience
        img_rows, img_cols = 28, 20
        ff = loadmat('../data/frey_rawface.mat', squeeze_me=True, struct_as_record=False)
        ff = ff["ff"].T.reshape((-1, 1, img_rows, img_cols))
        ff = ff.astype('float32')/255.

        size = len(ff)

        ff = ff[:int(size/b_size)*b_size]

        ff_torch = torch.from_numpy(ff)

        train_loader = torch.utils.data.DataLoader(ff_torch, b_size,
                                                   shuffle=True, **kwargs)
        
        test_loader = None

    elif data == 'cifar':
        train_loader = torch.utils.data.DataLoader(
            datasets.cifar.CIFAR10('../data', train=True, download=True,
                           transform=transforms.ToTensor()),
                            batch_size=b_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.cifar.CIFAR10('../data', train=False, download=True,
                           transform=transforms.ToTensor()),
                            batch_size=b_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def fetch_file(url,folder):
    """Downloads a file from a URL into a folder
    """
    try:
        f = urlopen(url)
        print("Downloading data file " + url + " ...")

        # Open our local file for writing
        with open(os.path.join(folder,os.path.basename(url)), "wb") as local_file:
            local_file.write(f.read())
        print("Done.")
    except:
        "Couldn't download data"


def check_frey():  
    url =  "http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat"
    data_filename = os.path.basename(url)
    if not os.path.exists(os.path.join('../data/',data_filename)):
        fetch_file(url,'../data/')
    else:
        print("Data file %s exists." % data_filename)


def train(model, optimizer, train_loader, loss_func, epochs = 1, show_prog = 100, summary = None):
    
    if summary:
        writer = SummaryWriter(summary)

    b_size = float(train_loader.batch_size)

            
    #writer.add_graph_onnx(model)
    
    model.train()
    for i in tqdm(range(epochs)):
        for batch_idx, (data) in enumerate(train_loader):

            if type(data) == list:
                data = data[0]

            n_iter = (i*len(train_loader))+batch_idx
            
            data = Variable(data, requires_grad = False)#.view(train_loader.batch_size,-1)  # NEED TO FLATTEN THE IMAGE FILE
            data = data.cuda()  # Make it GPU friendly
            optimizer.zero_grad()   # reset the optimzer so we don't have grad data from the previous batch
            dec_m, dec_v, enc_m, enc_v = model(data)   # forward pass
            loss = loss_func(enc_m, enc_v, data, dec_m, dec_v, model) # get the loss
            """
            if summary:
                # write the negative log likelihood ELBO per data point to tensorboard
                writer.add_scalar('ave loss/datapoint', -loss.data[0]/b_size, n_iter)
                w_s = torch.cat([torch.cat(layer.weight.data) for layer in model.children()]).abs().sum()
                writer.add_scalar('sum of NN weights', w_s, n_iter) # check for regularisation
            """
            loss.backward() # back prop the loss
            optimizer.step()    # increment the optimizer based on the loss (a.k.a update params)
            if batch_idx % show_prog == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
                if summary:
                    writer.add_image('real_image', data[1].view(-1,model.h,model.w), n_iter)
                    a,_,_,_ = model(data[1].unsqueeze(0).cuda())
                    #a, _, _, _ = model(data[1].unsqueeze(0))
                    writer.add_image('reconstruction', a.view(-1,model.h,model.w), n_iter)
                    b,_ = model.decode(model.sample().unsqueeze(0))
                    writer.add_image('from_noise', b.view(-1,model.h,model.w), n_iter)

# initialise the weights as per the paper
def init_weights(m):
    print("Messing with weights")
    #print(m)
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0,0.01)
