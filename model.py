import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class characterModel(nn.Module):
    def __init__(self,config):
        super(characterModel,self).__init__()

        self.config = config

        
