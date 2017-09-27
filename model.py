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
        self.vocab_size = config.vocab_size
        self.char_dim = config.char_dim
        self.batch_size = config.batch_size

        self.conv1 = nn.Conv3d(1,25,(1,self.char_dim,1))
        self.conv2 = nn.Conv3d(1,25*2,(1,self.char_dim,2))
        self.conv3 = nn.Conv3d(1,25*3,(1,self.char_dim,3))
        self.conv4 = nn.Conv3d(1,25*4,(1,self.char_dim,4))
        self.conv5 = nn.Conv3d(1,25*5,(1,self.char_dim,5))
        self.conv6 = nn.Conv3d(1,25*6,(1,self.char_dim,6))

        self.high_t = nn.Linear(525,525) 
        self.high_h = nn.Linear(525,525)
        
        self.lstm1 = nn.LSTM(525,300,batch_first=True)
        self.lstm2 = nn.LSTM(300,self.vocab_size,batch_first=True)
        self.softmax = nn.LogSoftmax()

    def forward(self,x):
        x1 = torch.max(F.tanh(self.conv1(x)),4)[0]
        x2 = torch.max(F.tanh(self.conv2(x)),4)[0]
        x3 = torch.max(F.tanh(self.conv3(x)),4)[0]
        x4 = torch.max(F.tanh(self.conv4(x)),4)[0]
        x5 = torch.max(F.tanh(self.conv5(x)),4)[0]
        x6 = torch.max(F.tanh(self.conv6(x)),4)[0]
        
        y = torch.squeeze(torch.cat((x1,x2,x3,x4,x5,x6),1)).transpose(1,2)
        t = F.relu(self.high_t(y))
        z = torch.mul(t,F.relu(self.high_h(y))) + torch.mul((1-t),y)
        
        lstm1_out, (h_1, c_1) = self.lstm1(z)
        lstm2_out,  (h_2, c_2) = self.lstm2(lstm1_out)
        lstm2_out = lstm2_out[:,-1,:] 
        ####################check softmax function (by row) 
        output = self.softmax(lstm2_out)
        
        return output
