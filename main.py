import numpy as np
import os.path
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.cuda as cutorch
import torch.optim as optim
from dataset import *
from util import *
from model import *
from run import *

class Config():
    def __init__(self):
        self.mode_list = ['train','valid','test'] 
        self.mode = 'test'
        self.data_dir = '/home/raehyun/github/characterNL/data'
        self.dict_name = ['tr_chdict.pickle','tr_wdict.pickle'] 
        
        self.more_train = False
        self.load_model = 're'
        self.save_model = 'retrain'

        self.char_dim = 15
        self.batch_size = 20
        self.vocab_size = 0
        self.time_step = 35
        
        self.lstm_hidden = 300
        self.num_layer = 2
        self.rnn_dropout = 0.5

        self.lr = 0.01
        self.tr_epoch = 10

def main():
    config = Config()
    
    char_dict,word_dict = GetDicts(config)    
    print(len(char_dict),len(word_dict)) 
   
    config.vocab_size = len(word_dict)
    
    lines = GetRawData(config.data_dir,config.mode)  # list of lines as a list of words
    
    model = CharNLM(config)
    model.cuda()
    # Put all words in one list 
    word_list,_ = wordToindex(word_dict,lines)
    input,target = get_batch(word_list,char_dict,word_dict,model.embedding,config) 
    
    #optimizer = optim.SGD(model.parameters(),lr=config.lr)
    #loss_function = nn.CrossEntropyLoss()
    
    run_train(input,target,model,config)
     
    if 'tr' in config.mode:
        if config.mode_train:
            model.load_state_dict(torch.load(
                '/home/raehyun/github/characterNL/model/%'%config.load_model))
        model.train()
        run_train(input,target,model,config)
    
if __name__ == "__main__":
    main()
