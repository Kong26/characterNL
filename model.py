import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class CharNLM(nn.Module):
    def __init__(self,config):
        super(CharNLM,self).__init__()

        self.char_dim = config.char_dim
        self.batch_size = config.batch_size
        self.time_step = config.time_step
        
        self.embedding = nn.Embedding(51,config.char_dim)
        
        self.char_cnn_fn = [1,2,3,4,5,6]
        self.char_conv = nn.ModuleList([nn.Conv2d(1,25*i,(self.char_dim,i))
                                        for i in self.char_cnn_fn])
        
        self.high_t = nn.Linear(525,525) 
        self.high_h = nn.Linear(525,525)
        
        self.lstm = nn.LSTM(525,config.lstm_hidden,config.num_layer,
                                dropout=config.rnn_dropout,batch_first=True)
        self.fc = nn.Linear(300,config.vocab_size)

        self.softmax = nn.Softmax()
    
    def char_conv_layer(self, inputs):
        conv_result = []
        for i, conv in enumerate(self.char_conv):
            out = torch.max(conv(inputs),3)[0].squeeze()
            conv_result.append(out)
        result = torch.cat(conv_result,1)
        
        return result
    
    def highway_layer(self,inputs,batch_size,time_step):
        t = F.relu(self.high_t(inputs))
        z = torch.mul(t,F.relu(self.high_h(inputs))) + torch.mul((1-t),inputs)
        z = z.view(batch_size,time_step,-1) 
        return z

    def lstm_layer(self,inputs,hiddens):
         out,(h,c) = self.lstm(inputs,hiddens)
         hidden = (h.data,c.data)
         return out,hidden
    
    def fc_layer(self,inputs):
        return self.fc(inputs)

    def forward(self,input,hidden):
        
        char_conv = self.char_conv_layer(input)
        z = self.highway_layer(char_conv,self.batch_size,self.time_step) 
        lstm_out,hidden = self.lstm_layer(z,hidden) 
        out = self.fc_layer(lstm_out)

        return out,hidden 

