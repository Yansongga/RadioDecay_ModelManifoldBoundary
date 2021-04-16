import torch 
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  
from model import *
import torch.optim as optim
from autograd_hacks import *
from autograd_hacks_test import *
args = {
    'dev': torch.device('cuda: 2' ),
    'trainsize': 10,
    'testsize': 100,
    'steps': 100000,
    't_range': 10,
    'mode': 8, 
    'lr': 1e-3,
    'alpha': 0.
}


#sampling function 
def real(t):
    a = torch.tensor( [ [0.2],[0.3],[0.1],[0.5] ] ).to(args['dev'])
    l = torch.tensor( [ [0.2],[0.21],[0.7],[0.19] ] ).to(args['dev'])
    h = ( - torch.matmul( t, l.T) ).exp()
    return torch.matmul( h, a )

t1, t2 = args['t_range'] * torch.rand(args['trainsize'], 1).to(args['dev']), args['t_range'] * torch.rand(args['testsize'], 1).to(args['dev'])

#generating train and test set
trainT = t1 + 0. 
trainY = real( trainT )

testT = t2 + 0. 
testY = real( testT )

#initialize model 
params = torch.rand(args['mode'], 2).to(args['dev'])
A, l = (params[:, 0] + 0.).unsqueeze(1), (params[:, 1] + 0.).unsqueeze(1)


#define functions, gradients(per sample), hessian 
# f has size N x 1
def f( x, A, l ):
    h = ( - torch.matmul( x, l.T) ).exp()
    return torch.matmul( h, A )

# loss is a scale 
def lossFunction(y, f):
    square = 0.5 * ( ( y - f )**2 )
    return square.sum()

# Agrad, lgrad  has size N x p, per sample gradient
def fGradient(x, A, l):
    Agrad = ( - torch.matmul( x, l.T) ).exp()
    lgrad = (- torch.matmul( x, A.T) ) * Agrad
    return Agrad, lgrad  

def lossGradient( y, f, Agrad, lgrad ):
    hA = torch.matmul( (y -f).T, Agrad)
    hl = torch.matmul( (y -f).T, lgrad)
    return -hA.T, -hl.T

def hessian( x, A, l, v ):
    h = ( - torch.matmul( x, l.T) ).exp()
    AA_grad = torch.matmul( v.T, h * 0. ).T
    Al_grad = torch.matmul(v.T, - x * h ).T
    ll_grad = ( torch.matmul( x**2, A.T) ) * h
    ll_grad = torch.matmul(v.T, ll_grad ).T
    hes = torch.cat( ( AA_grad, Al_grad, Al_grad, ll_grad), 1 )
    return hes.view( args['mode'], 2, 2 )

def prod(hes, Agrad, lgrad, v):
    vAgrad, vlgrad = torch.matmul( v.T, Agrad), torch.matmul( v.T, lgrad)
    vec = torch.cat( (vAgrad.T, vlgrad.T), 1 ).unsqueeze(2)
    prod = torch.matmul( hes, vec )
    return prod[:, 0, :], prod[:, 1, :]
    
def kernel_decomp( Agrad, lgrad ):
    kernel = torch.matmul( Agrad, Agrad.T) + torch.matmul( lgrad, lgrad.T)
    w, v = torch.symeig(kernel, eigenvectors=True)
    k = w.size()[0] - 3
    return w[k], v[:, k].unsqueeze(1)





###train with regulizer
for step in range(args['steps']):
    trainF = f( trainT, A, l ) 
    Agrad, lgrad = fGradient(trainT, A, l)
    w, v = kernel_decomp( Agrad, lgrad )
    hes = hessian( trainT, A, l, v )
    
    #gradient with respect to loss and eigenvalue
    dA, dl = prod(hes, Agrad, lgrad, v)
    loss_A, loss_l = lossGradient( trainY, trainF, Agrad, lgrad )
    
    #gradiend descent
    A -= args['lr'] * ( loss_A / args['trainsize'] + args['alpha'] * dA )
    l -= args['lr'] * ( loss_l / args['trainsize'] + args['alpha'] * dl )
    
    if (step + 1) % 2500 == 0 or step == 0:    # print every 1000 steps
        testF = f( testT, A, l ) 
        trainloss = lossFunction(trainY, trainF)
        testloss = lossFunction(testY, testF)
        
        print('[%5d]  eigenvalue: %.5f  trainloss: %.5f  testloss: %.5f' %
              (step + 1, w, trainloss, testloss/10  ))
        
