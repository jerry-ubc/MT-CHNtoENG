# import torch
# import torch.optim as optim
# import torch.nn as nn
# import spacy    #vocabulary and tokenizer
# import spacy
# from transformer_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
# from torch.utils.tensorboard import SummaryWriter   #loss plots
# from torchtext.datasets import Multi30k
# from torchtext.data import Field, BucketIterator

# spacy_zh = spacy.load('zh_core_web_trf')
# spacy_en = spacy.load('en_core_web_trf')

# #tokenizers
# def tokenize_zh(text):
#     return [tok.text for tok in spacy_zh.tokenizer(text)]
# def tokenize_en(text):
#     return [tok.text for tok in spacy_en.tokenizer(text)]

# #tokenizer parameters
# chinese = Field(tokenize=tokenize_zh, lower=True, init_token='<sos>', eos_token='<eos>')
# english = Field(tokenize=tokenize_en, lower=True, init_token='<sos>', eos_token='<eos>')

# #train_data, dev_data, test_data = Multi30k.splits(exts=('.zh', '.en'), fields=(chinese, english))
# #RUN corpus_process.py FIRST!
# #train_data = open(zhen_data/)


# # chinese.build_vocab(train_data, max_size=10000, min_freq=2)
# # english.build_vocab(train_data, max_size=10000, min_freq=2)


from fnmatch import translate
import torch
import torch.optim as optim
import torch.nn as nn
import spacy    #vocabulary and tokenizer
from transformer_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter   #loss plots
from torchtext.datasets import Multi30k     #German -> English dataset
from torchtext.data import Field, BucketIterator    #data processing

spacy_ge = spacy.load('de_core_news_md')
spacy_en = spacy.load('en_core_web_md')

#tokenizers
def tokenize_ge(text):
    return [tok.text for tok in spacy_ge.tokenizer(text)]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

#tokenize datasets
german = Field(tokenize=tokenize_ge, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenize_en, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, dev_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

print(train_data)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)