import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from util import *
from model import *

class Config():
    def __init__(self):
        self.mode_list = ['train','valid','test'] 
        self.mode = 'train'
        self.data_dir = '/home/raehyun/github/characterNL/data'
        
        self.char_dim = 15
        self.batch_size = 20
        self.vocab_size = 0

def main():
    config = Config()
    
    char_dict, word_dict, max_w =  data_statics(config.data_dir,config.mode_list)
    config.vocab_size = len(word_dict)
    
    file = get_data_file(config.data_dir,config.mode) 
    lines = create_line_list(file)  # list of lines as a list of words
    
    embedding = nn.Embedding(len(char_dict),config.char_dim) 

    indexed_list = charToindex(char_dict,lines)
    indexed_list.sort(key=len,reverse=True)
    max_s = len(indexed_list[0])
    
    lines_tensor, lines_mask = make_sent_vector(indexed_list[:config.batch_size],embedding,max_s) 
    
    
    model = CharNLM(config)
    
    for w_ix in range(max_s): 
        input = Variable(lines_tensor[:,w_ix,:,:],requires_grad=False).contiguous().view([config.batch_size,1,config.char_dim,max_w])
    
        output = model(input)
        print(output.data.shape) 
        break

if __name__ == "__main__":
    main()
