import numpy as np
import os.path
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
from util import *
from model import *
import torch.cuda as cutorch
import gpustat
import torch.optim as optim
class Config():
    def __init__(self):
        self.mode_list = ['train','valid','test'] 
        self.mode = 'train'
        self.data_dir = '/home/raehyun/github/characterNL/data'
        self.dict_name = ['tr_chdict.pickle','tr_wdict.pickle'] 
        self.split_num = 10
        

        self.char_dim = 15
        self.batch_size = 20
        self.vocab_size = 0
        self.time_step = 35
        self.lstm_hidden = 300

        self.lr = 0.01
        self.tr_epoch = 10

def show_gpu(device=1):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item['memory.used'],item["memory.total"]))

def main():
    config = Config()
    
    # Create or load char & word dictionaries
    if os.path.isfile(config.dict_name[0]):
        with open(config.dict_name[0],'rb') as char:
            char_dict = pickle.load(char)
        with open(config.dict_name[1],'rb') as word:
            word_dict = pickle.load(word)
    else:
        char_dict, word_dict, max_w =  data_statics(config.data_dir,config.mode_list)
        with open(config.dict_name[0],'wb') as char:
            pickle.dump(char_dict,char,protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.dict_name[1],'wb') as word:
            pickle.dump(word_dict,word,protocol=pickle.HIGHEST_PROTOCOL)
    
    config.vocab_size = len(word_dict)
    
    file = get_data_file(config.data_dir,config.mode) 
    lines = create_line_list(file)  # list of lines as a list of words

    embedding = nn.Embedding(len(char_dict),config.char_dim) 
    # Put all words in one list 
    word_list,_ = wordToindex(word_dict,lines)
    batch_length = len(word_list)//config.split_num

    print("before model")
    show_gpu()    

    model = CharNLM(config)
    print(model)
    model.cuda()
    print("after model")
    show_gpu()    
    
    
    optimizer = optim.SGD(model.parameters(),lr=config.lr)
    loss_function = nn.NLLLoss()
    
    if 'tr' in config.mode:
        for epoch in range(config.tr_epoch):
            for split_ix in range(config.split_num):
                batch_line = word_list[split_ix*batch_length:split_ix*batch_length+batch_length]

                input,target = get_batch(batch_line,char_dict,word_dict,embedding,config.batch_size,config.time_step)
                
                for ix in range(input.shape[2]//config.time_step):
                    optimizer.zero_grad()
                    
                    m_input = Variable(input[:,:,ix*config.time_step:ix*config.time_step+config.time_step,:,:],requires_grad=False).cuda()
                    m_target = Variable(target[:,ix*config.time_step+1:ix*config.time_step+config.time_step],requires_grad=False).contiguous().view((config.batch_size*(config.time_step-1))).cuda()
                    
                    print(m_input.data.shape,m_target.data.shape)
                    
                    print("after input")
                    show_gpu()    
                    
                    output = model(m_input).view((config.batch_size*(config.time_step-1),-1))
                    print("after model call")
                    show_gpu()    
                    
                    loss = loss_function(output,m_target)
                    #loss = torch.exp((loss_function(output,target)*config.batch_size)/(config.time_step-1))
                    print("after get loss")
                    show_gpu() 
                    
                    if ix%10 == 0:
                        nll = loss.data
                        ppl = torch.exp(nll/(config.time_step-1))
                        print('loss of training step %i : %f'%(ix,ppl[0]))
                    
                    loss.backward()
                    print("after backward")
                    show_gpu() 
                    
                    optimizer.step()
                    print("after optimizer step")
                    show_gpu()    

if __name__ == "__main__":
    main()
