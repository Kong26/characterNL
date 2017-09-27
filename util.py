import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

def find_longest(lines):
    max_length = 0
    max_word = 'dummy'
    
    for line in lines:
        for word in line:
            if len(word)>max_length:
                max_length = len(word)
                max_word = word
    return max_length

def get_data_file(data_dir,mode):
    data = os.path.join(data_dir,'%s.txt'%mode)
    f = open(data,'r')
    return f

def create_line_list(file):
    lines = file.readlines()
    line_list = []
    for line in lines:
        line_list.append(line.split())
    return line_list

def make_embedding(length,dim):
    return nn.Embedding(length,dim)

def make_char_dict(lines):
    total_character = set()
    for line in lines:
        for word in line:
            for char in word:
                total_character.add(char)
    char_dict = {}
    for idx,char in enumerate(total_character):
        char_dict[char] = idx
    return char_dict

def make_word_dict(lines):
    total_word = set() 
    for line in lines:
        for word in line:
            total_word.add(word)
    word_dict = {}
    for idx,word in enumerate(total_word):
        word_dict[word] = idx
    return word_dict

def data_statics(data_dir,mode_list):
    for mode in mode_list:
        file = get_data_file(data_dir,mode)
        lines = create_line_list(file)
        char_dict = make_char_dict(lines)
        word_dict = make_word_dict(lines)
        if 'tr' in mode:
            max_w = find_longest(lines)
            out_char = char_dict
            out_word = word_dict
        print(mode,'data statics')
        print('char dict length : %i'%len(char_dict))
        print('word dict length : %i'%len(word_dict))
    
    return out_char,out_word,max_w

def charToindex(char_dict,lines):
    line_list = []
    for line in lines:
        idx_line = []
        for word in line:
            word_idx = []
            for idx,str in enumerate(word):
                word_idx.append(char_dict[str])
            idx_line.append(word_idx)
        line_list.append(idx_line)
    return line_list

def wordToindex(word_dict,lines):
    index_list = []
    word_list = []
    for line in lines:
        for word in line:
            index_list.append(word_dict[word])
            word_list.append(word)
    return word_list,index_list

def indexTovector(lines,embedding,time_step):
    batch_size = len(lines) 
    vector_list = []
    batch_tensor = torch.zeros(batch_size,time_step,15,19)

    for ix_l,line in enumerate(lines): 
        sent_tensor = torch.zeros(time_step,15,19) 
        for ix_w,word in enumerate(line): 
            word_vector = torch.zeros(15,19)
            for i,ix in enumerate(word):
                input = Variable(torch.LongTensor([ix]))
                word_vector[:,i] = embedding(input).data
            sent_tensor[ix_w] = word_vector
        batch_tensor[ix_l] = sent_tensor
    return batch_tensor

'''def make_sent_vector(lines_list,embedding,max_s):
    # create tensor size of (Batch_size, Max_s, char_dim, Max_w)
    
    lines_tensor = torch.zeros(len(lines_list),max_s,15,19)
    lines_mask = torch.zeros(len(lines_list),max_s) 
    
    for ix_l,line in enumerate(lines_list):
        lines_tensor[ix_l] = indexTovector(line,embedding,max_s)
        mask_vector = torch.ones(len(line))
        lines_mask[ix_l][:len(line)] = 1
    return lines_tensor,lines_mask'''

def make_sent_vector(word_batch, embedding, batch_size):
    # create tensor size of (batch, # of time step we see, char_dim, max_w)
    lines_tensor = torch.zeros(batch_size,word_batch.shape[1]-1,15,19)
    
    for ix_l,line in enumerate(lines_list):
        lines_tensor[ix_l] = indexTovector(line,embedding,word_batch.shape[1]-1)
    
    return lines_tensor


def create_batch(word_list,start,batch_size):
    word_batch = []
    for ix in range(start,start+batch_size):
        word_batch.append(word_list[start+ix:start+35+ix])
    return word_batch

def get_next_batch(word_list, char_dict, word_dict, embedding, start, batch_size, time_step):
    # word_batch : List of batches - batch size * # of words
    word_batch = create_batch(word_list, start, batch_size)
    indexed_batch = charToindex(char_dict,word_batch)
    
    input_words = indexTovector(indexed_batch,embedding,time_step)[:,:time_step-1,:,:] 
    target_words = torch.LongTensor([word_dict[line[-1]] for line in word_batch])
    
    return input_words, target_words





















