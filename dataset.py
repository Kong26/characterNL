import numpy as np 
from util import *

def GetRawData(data_dir,mode):
    file = get_data_file(data_dir,mode)
    # list of lines as a list of words
    lines = create_line_list(file)

    return lines
