import numpy as np
from dataset import *
from util import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import *
import torch.optim as optim
import torch.cuda as cutorch

def init_hidden(num_layer,batch_size,lstm_hidden):
    h = Variable(torch.zeros(num_layer,batch_size,lstm_hidden)).cuda()
    c = Variable(torch.zeros(num_layer,batch_size,lstm_hidden)).cuda()
    return (h,c)

def run_train(t_input,t_target,model,config):
    
    t = config.time_step 
    optimizer = optim.SGD(model.parameters(),lr=config.lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(config.tr_epoch):
        
        for ix in range(t_input.shape[2]//config.time_step):
            
            optimizer.zero_grad()

            #m_input = (20*35,15,21)
            m_input = Variable(t_input[:,:,ix*t:ix*t+t,:,:].contiguous().view(
                [t*config.batch_size,1,15,21]),requires_grad=False).cuda()
            #m_target = (20,35)
            m_target = Variable(t_target[:,ix*t+1:ix*t+t+1],requires_grad=False)\
                    .contiguous().view((config.batch_size*(t))).cuda()
            
            if ix == 0:
                hidden = init_hidden(config.num_layer,config.batch_size,
                                                            config.lstm_hidden)
            else:
                h0,c0 = hidden
                h0 = Variable(h0).cuda()
                c0 = Variable(c0).cuda()
                hidden = (h0,c0)
            
            out,hidden = model(m_input,hidden)
            out = out.contiguous().view([config.batch_size*config.time_step,-1]) 
            
            loss = loss_function(out,m_target)
            
            loss.backward() 
            optimizer.step()
            
            if ix%100 == 0:
                nll = loss.data
                ppl = torch.exp(nll)
                print('loss of training epoch %i step %i : %f'%(epoch,ix,ppl[0]))
            
    torch.save(model.state_dict(),
            '/home/raehyun/github/characterNL/model/%s'%config.save_model)
