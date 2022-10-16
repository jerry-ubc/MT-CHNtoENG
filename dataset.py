import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from zhutils import english_tokenizer_load
from zhutils import chinese_tokenizer_load
import csv

class NCDataset(Dataset):
    # def __init__(self, data_path):
    #     self.out_en_sents, self.out_zh_sents = self.get_dataset(data_path)
    #     self.spc_en = english_tokenizer_load()
    #     self.spc_zh = chinese_tokenizer_load()

    def __init__(self, src_data_path, tgt_data_path):
        self.out_en_sents = self.get_dataset(src_data_path)
        self.out_zh_sents = self.get_dataset(tgt_data_path)
        self.spc_en = english_tokenizer_load()
        self.spc_zh = chinese_tokenizer_load()
        self.PAD = self.spc_en.pad_id()     #0
        self.BOS = self.spc_en.bos_id()     #2
        self.EOS = self.spc_en.eos_id()     #3

    def get_dataset(self, data_path):
        out_sentences = []
        # if is_source:
        #     data_path = '{}.zh'.format(data_path)
        # else:
        #     data_path = '{}.en'.format(data_path)
        #print("datapath: " + data_path)
        with open(data_path, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                out_sentences.append(line)
        file_in.close()
    

    # def get_dataset(self, data_path):
    #     out_en_sentences = []
    #     out_zh_sentences = []

    #     #populate lists of sentences
    #     with open(data_path, 'r', encoding='utf-8') as file_in:
    #         for line in file_in:
    #             out_en_sentences.append(line[0] + '\n')
    #             out_zh_sentences.append(line[1] + '\n')
        
    #     return out_en_sentences, out_zh_sentences

    def __getitem__(self, idx):
        en_sent = self.out_en_sents[idx]
        zh_sent = self.out_zh_sents[idx]
        return [en_sent, zh_sent]

    def __len__(self):
        return len(self.out_en_sents)