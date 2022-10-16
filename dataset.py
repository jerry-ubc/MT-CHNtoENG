import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from zhutils import english_tokenizer_load
from zhutils import chinese_tokenizer_load

class NCDataset(Dataset):
    def __init__(self, en_data_path, zh_data_path):
        self.out_en_sent = self.get_dataset(en_data_path)
        self.out_zh_sent = self.get_dataset(zh_data_path)
        self.spc_en = english_tokenizer_load()
        self.spc_zh = chinese_tokenizer_load()

    def get_dataset(self, en_data_path, zh_data_path):
        pass