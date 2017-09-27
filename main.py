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
        
        self.char_dim = 15
        self.batch_size = 20
        self.vocab_size = 0
        self.time_step = 36
        
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
    
    print("before model")
    show_gpu()    

    model = CharNLM(config)
    model.cuda()
    print("after model")
    show_gpu()    

    optimizer = optim.SGD(model.parameters(),lr=config.lr)
    loss_function = nn.CrossEntropyLoss()
    
    if 'tr' in config.mode:
        for epoch in range(config.tr_epoch):
            for ix in range(len(word_list)-config.batch_size): 
                start = ix * config.batch_size
                input, target = get_next_batch(word_list,char_dict,word_dict,embedding,start,config.batch_size,config.time_step) 
                
                input, target = Variable(input.contiguous().view([config.batch_size,1,config.time_step-1,15,19]),requires_grad=False).cuda(), \
                                                                                    Variable(target,requires_grad=False).cuda()
                print("after input")
                show_gpu()    
                
                output = model(input)
                optimizer.zero_grad()
                print("after model call")
                show_gpu()    
                
                loss = loss_function(output,target)
                #loss = torch.exp((loss_function(output,target)*config.batch_size)/(config.time_step-1))
                print("after get loss")
                show_gpu() 
                
                #if ix%10 == 0:
                 #   print('loss of training step %i : %f'%(ix,loss.data[0]))
                
                loss.backward()
                print("after backward")
                show_gpu() 
                
                optimizer.step()
                print("after optimizer step")
                show_gpu()    
 

if __name__ == "__main__":
    main()
