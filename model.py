import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class CharNLM(nn.Module):
    def __init__(self,config):
        super(CharNLM,self).__init__()

        self.config = config
        self.char_dim = config.char_dim
        self.batch_size = config.batch_size

        self.conv1 = nn.Conv2d(1,25,(self.char_dim,1))
        self.conv2 = nn.Conv2d(1,25*2,(self.char_dim,2))
        self.conv3 = nn.Conv2d(1,25*3,(self.char_dim,3))
        self.conv4 = nn.Conv2d(1,25*4,(self.char_dim,4))
        self.conv5 = nn.Conv2d(1,25*5,(self.char_dim,5))
        self.conv6 = nn.Conv2d(1,25*6,(self.char_dim,6))

        self.high_t = nn.Linear(525,525) 
        self.high_h = nn.Linear(525,525)
        
        self.lstm1 = nn.LSTM(525,300)
        self.lstm2 = nn.LSTM(300,config.vocab_size)

    def forward(self,x):
        x1 = torch.max(F.tanh(self.conv1(x)),3)[0]
        x2 = torch.max(F.tanh(self.conv2(x)),3)[0]
        x3 = torch.max(F.tanh(self.conv3(x)),3)[0]
        x4 = torch.max(F.tanh(self.conv4(x)),3)[0]
        x5 = torch.max(F.tanh(self.conv5(x)),3)[0]
        x6 = torch.max(F.tanh(self.conv6(x)),3)[0]
        
        y = torch.squeeze(torch.cat((x1,x2,x3,x4,x5,x6),1),2)
        t = F.relu(self.high_t(y))
        z = torch.mul(t,F.relu(self.high_h(y))) + torch.mul((1-t),y)
        
        z = z.view(1,self.batch_size,525)

        h_1, c_1 = self.lstm1(z)
        h_2, c_2 = self.lstm2(h_1)
        
        return h_2
