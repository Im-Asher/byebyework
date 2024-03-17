import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, emb_size, eps=1e-8, dim=3) -> None:
        super(Dice, self).__init__()
        self.name = 'Dice'
        self.dim = dim
        self.dimJudge(dim, 2, 3)
        self.bn = nn.BatchNorm1d(emb_size, eps=eps)
        self.sig = nn.Sigmoid()
        if dim == 2:  # [B,C]
            self.alpha = torch.zeros((emb_size,))
            self.beta = torch.zeros((emb_size,))
        elif dim == 3:  # [B,C,E]
            self.alpha = torch.zeros((emb_size, 1))
            self.beta = torch.zeros((emb_size, 1))

    def forward(self, x):
        if self.dim == 2:
            x_n = self.sig(self.beta * self.bn(x))
            return self.alpha * (1-x_n) * x + x_n * x
        elif self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_n = self.sig(self.beta * self.bn(x))
            output = self.alpha * (1-x_n) * x + x_n * x
            output = torch.transpose(output, 1, 2)
            return output
    
    def dimJudge(dim1,dim2,dim3):
        assert dim1 == dim2 or dim1 == dim3, 'dimension is not correct'

class PRelu(nn.Module):
    def __init__(self,size):
        super(PRelu,self).__init__()
        self.name = 'Prelu'
        self.alpha = torch.zeros((size,))
        self.relu = nn.Relu()
        
    def forward(self,x):
        pos = self.relu(x) #only for positive part
        neg = self.alpha * (x - abs(x)) * 0.5 #only for negetive part
        return pos + neg

class ActivationUnit(nn.Module):
    def __init__(self,inSize,af='dice',hidden_size=36):
        super(ActivationUnit,self).__init__()
        self.name = 'activation_unit'
        self.linear1 = nn.Linear(inSize,hidden_size)
        self.linear2 = nn.Linear(hidden_size,1)
        if af == 'dice':
            self.af = Dice(hidden_size,dim=2)
        elif af == 'prelu':
            self.af = PRelu()
        else:
            print('only dice and prelu can be chosen for activation function')
        
    def forward(self,item1,item2): #[B,C]
        cross = torch.mm(item1,item2.T)
        x = torch.cat([item1,cross,item2],-1) #[B,B+2*C]
        x = self.linear1(x)
        x = self.af(x)
        x = self.linear2(x)
        return x