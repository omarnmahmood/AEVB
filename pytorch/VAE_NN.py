import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.init import xavier_normal
from torchvision import datasets, transforms
from tqdm import tqdm   # Progress bar
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import os
from urllib.request import urlopen
from scipy.io import loadmat
import pdb

from tensorboardX import SummaryWriter

#from torchvision.utils import save_image

class VAE_Net(nn.Module):
    
    # MAIN VAE Class

    def __init__(self, latent_size=20, data='MNIST', conditional = False, fast = False):
        super(VAE_Net, self).__init__()

        # if fast mode is on then let's train faster

        if fast:
            self.act_f = F.relu
        else:
            self.act_f = F.tanh

        # define the encoder and decoder
        
        self.data = data
        self.conditional = conditional

        if self.conditional:
            self.cond_s = 10
        else:
            self.cond_s = 0
        
        if (self.data == 'Frey') & (self.conditional == True):
            raise ValueError('No classes in Frey, cannot do a conditional VAE on this data (yet)')

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

        self.latent = latent_size

        self.ei = nn.Linear(self.h * self.w * 1 + self.cond_s, self.u)
        self.em = nn.Linear(self.u, self.latent)
        self.ev = nn.Linear(self.u, self.latent)

        self.di = nn.Linear(self.latent + self.cond_s, self.u)
        self.dom = nn.Linear(self.u, self.h * self.w * 1)


    def encode(self, x):

        # encoder part

        o = self.act_f(self.ei(x))
        mu = self.em(o)
        logvar = self.ev(o)
        return mu, logvar

    def decode(self, x):

        # decoder part    
        
        # THIS IS THE ORIGINAL
        #o = F.sigmoid(self.di(x))
        #im = F.sigmoid(self.do(o))
        #return im

        o = self.act_f(self.di(x))
        im = F.sigmoid(self.dom(o))
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
        decode_info = self.repar(mu,logvar)
        if self.conditional:
            label_ohc = x[:,-10:]
            decode_info = torch.cat([decode_info, label_ohc], dim = 1)
        om, ov = self.decode(decode_info)
        return om, ov , mu, logvar

def elbo_loss(enc_m, enc_v, x, dec_m, dec_v, model, beta = 1):

    # ELBO/Loss

    size = enc_m.size()

    KL_part = 0.5*((enc_v.exp().sum() + enc_m.dot(enc_m) - size[0]*size[1] - enc_v.sum()))
    
    if model.data == 'Frey':    # get the complicated Gaussian reconstruction term

        Recon_part = torch.sum(    torch.sum(    ((x - dec_m)**2)*(1./dec_v.exp()),dim=1    )   )

        Pi_term = Variable((0.5 * torch.Tensor([2*np.pi]).log() * size[0]).cuda())
        
        Recon_norm = torch.sum(0.5 * torch.sum(dec_v, dim = 1))
        
        Recon_total = Recon_part + Recon_norm + Pi_term
    
    else:   # assume MNIST, therefore 'pseudo-binary' and use Bernoulli
        
        Recon_total = F.binary_cross_entropy(dec_m, x, size_average=False)
    
    #print('Recon loss: %.6f' % (Recon_part))
    #print('KL loss: %.6f' % KL_part)
    #print('MSE loss: %.6f' % MSE_part)
    #print('Variance sum: %.6f' % dec_v.sum())

    output = (Recon_total + beta * KL_part) # ave loss per datapoint

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


def train(model, optimizer, train_loader, loss_func, epochs = 1, show_prog = 100, summary = None, test_loader = None, scheduler = None, beta = 1):
    
    if summary:
        writer = SummaryWriter()

    ohc = OneHotEncoder(sparse=False)
    
    # fit to some dummy data to prevent errors later
    ohc.fit(np.arange(0,10).reshape(10,1))

    b_size = float(train_loader.batch_size)
            
    #writer.add_graph_onnx(model)
    
    # add an initial values for the likelihoods to prevent weird glitches in tensorboard
    if summary:
        if test_loader:
            test_loss = get_loss(model, test_loader, loss_func, ohc)
            writer.add_scalar('loss/ave_test_loss_per_datapoint', -test_loss, 0)
        train_loss = get_loss(model, train_loader, loss_func, ohc)
        writer.add_scalar('loss/ave_loss_per_datapoint', -train_loss, 0)
    
    model.train()
    for i in tqdm(range(epochs)):
        if scheduler:
            scheduler.step()
            print(optimizer.state_dict()['param_groups'][0])
        for batch_idx, (data) in enumerate(train_loader):

            if type(data) == list:
                label = data[1]
                data = data[0]

            #n_iter = (i*len(train_loader))+batch_idx
            n_iter = (i*len(train_loader)*b_size) + batch_idx*b_size

            data = Variable(data.view(train_loader.batch_size,-1), requires_grad = False)
            if model.conditional:
                label = Variable(torch.Tensor(ohc.transform(label.numpy().reshape(len(label), 1))))
                data = torch.cat([data,label],dim=1)
            data = data.cuda()  # Make it GPU friendly
            optimizer.zero_grad()   # reset the optimzer so we don't have grad data from the previous batch
            dec_m, dec_v, enc_m, enc_v = model(data)   # forward pass
            if model.conditional:
                data_o = data[:,:-10]
            else:
                data_o = data
            loss = loss_func(enc_m, enc_v, data_o, dec_m, dec_v, model, beta) # get the loss
            if summary:
                # write the negative log likelihood ELBO per data point to tensorboard
                #pdb.set_trace()
                writer.add_scalar('loss/ave_loss_per_datapoint', -loss.data[0]/b_size, n_iter)
                #w_s = torch.cat([torch.cat(layer.weight.data) for layer in model.children()]).abs().sum()
                #writer.add_scalar('sum of NN weights', w_s, n_iter) # check for regularisation
            loss.backward() # back prop the loss
            optimizer.step()    # increment the optimizer based on the loss (a.k.a update params)
            if batch_idx % show_prog == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
                if summary:
                    writer.add_image('real_image', data_o[1].view(-1,model.h,model.w), n_iter)
                    a,_,_,_ = model(data[1:2].cuda())
                    writer.add_image('reconstruction', a.view(-1,model.h,model.w), n_iter)
                    if model.conditional:
                        p = np.random.randint(0,10)
                        num = [0]*10
                        num[p] = 1
                        num = Variable(torch.Tensor(num)).cuda()
                        b,_ = model.decode(torch.cat([model.sample(),num]))
                    else:
                        b,_ = model.decode(model.sample())
                    writer.add_image('from_noise', b.view(-1,model.h,model.w), n_iter)
        if test_loader and summary:
            test_loss = get_loss(model, test_loader, loss_func, ohc)
            writer.add_scalar('loss/ave_test_loss_per_datapoint', -test_loss, n_iter + b_size)

# return the loss over a dataset held in a dataloader
def get_loss(model, data_loader, loss_func, one_hot_encoder):

    b_size = float(data_loader.batch_size)

    loss = []    

    for batch_idx, (data) in enumerate(data_loader):
        
        if type(data) == list:
            label = data[1]
            data = data[0]

        data = Variable(data.view(data_loader.batch_size,-1), requires_grad = False)
        if model.conditional:
            label = Variable(torch.Tensor(one_hot_encoder.transform(label.numpy().reshape(len(label), 1))))
            data = torch.cat([data,label],dim=1)
        data = data.cuda()  # Make it GPU friendly
        dec_m, dec_v, enc_m, enc_v = model(data) # get the outputs based on test
        if model.conditional:
            data = data[:,:-10]
        loss.append(loss_func(enc_m, enc_v, data, dec_m, dec_v, model).data[0])

    loss = np.mean(loss)/b_size

    return loss

# initialise the weights as per the paper
def init_weights(m):
    print("Messing with weights")
    #print(m)
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0,0.01)

def init_weights_xavier(m):
    print("Messing with weights Xavier")
    if type(m) == nn.Linear:
        xavier_normal(m.weight)
