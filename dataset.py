import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchtext.datasets import TranslationDataset
from torch.nn.utils.rnn import pad_sequence
from zhutils import english_tokenizer_load
from zhutils import chinese_tokenizer_load
import csv
import os

class NCDataset(TranslationDataset):
    """Custom Dataset for NCv16"""

    urls = ['http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/'
            'wmt17_files_mmt/mmt_task1_test2016.tar.gz']
    name = 'ncv16'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of the NCv16 dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(NCDataset, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


class WTDataset(TranslationDataset):
    """Custom Dataset for WTv2"""

    urls = ['http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/'
            'wmt17_files_mmt/mmt_task1_test2016.tar.gz']
    name = 'wtv2'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train_small', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of the Wiki Titles v2 dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']
        print("path: " + path)
        return super(WTDataset, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


# class NCDataset(Dataset):
# class NCDataset(TranslationDataset):
#     # def __init__(self, data_path):
#     #     self.out_en_sents, self.out_zh_sents = self.get_dataset(data_path)
#     #     self.spc_en = english_tokenizer_load()
#     #     self.spc_zh = chinese_tokenizer_load()

#     def __init__(self, src_data_path, tgt_data_path):
#         self.out_zh_sents = self.get_dataset(src_data_path)
#         self.out_en_sents = self.get_dataset(tgt_data_path)
#         self.spc_zh = chinese_tokenizer_load()
#         self.spc_en = english_tokenizer_load()
#         self.PAD = self.spc_en.pad_id()     #0
#         self.BOS = self.spc_en.bos_id()     #2
#         self.EOS = self.spc_en.eos_id()     #3

#     def get_dataset(self, data_path):
#         out_sentences = []
#         # if is_source:
#         #     data_path = '{}.zh'.format(data_path)
#         # else:
#         #     data_path = '{}.en'.format(data_path)
#         #print("datapath: " + data_path)
#         with open(data_path, 'r', encoding='utf-8') as file_in:
#             #print("opened: " + data_path)
#             for line in file_in:
#                 out_sentences.append(line)
#                 #print("appended: " + line)
#         file_in.close()
#         return out_sentences
    

    # def get_dataset(self, data_path):
    #     out_en_sentences = []
    #     out_zh_sentences = []

    #     #populate lists of sentences
    #     with open(data_path, 'r', encoding='utf-8') as file_in:
    #         for line in file_in:
    #             out_en_sentences.append(line[0] + '\n')
    #             out_zh_sentences.append(line[1] + '\n')
        
    #     return out_en_sentences, out_zh_sentences

    # def __getitem__(self, idx):
    #     en_sent = self.out_en_sents[idx]
    #     zh_sent = self.out_zh_sents[idx]
    #     return [en_sent, zh_sent]

    # def __len__(self):
    #     return len(self.out_en_sents)