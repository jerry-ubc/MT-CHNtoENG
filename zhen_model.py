import torch
import torch.optim as optim
import torch.nn as nn
import spacy    #vocabulary and tokenizer
import spacy
from transformer_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter   #loss plots
from torchtext.datasets import Multi30k, NCDataset
from torchtext.data import Field, BucketIterator
from spacy.lang.zh import Chinese

spacy_zh = spacy.load('zh_core_web_trf')        #MAYBE CAN COMMENT OUT!!!!!
spacy_en = spacy.load('en_core_web_trf')



# Jieba
cfg = {"segmenter": "jieba"}
nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})

#tokenizers
def tokenize_zh(text):
    return nlp(text)
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


#tokenizer parameters
chinese = Field(tokenize=tokenize_zh, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenize_en, lower=True, init_token='<sos>', eos_token='<eos>')

#train_data, dev_data, test_data = Multi30k.splits(exts=('.zh', '.en'), fields=(chinese, english))
#RUN corpus_process.py FIRST!
#train_data = open(zhen_data/)
train_data, dev_data, test_data = NCDataset(exts = ('.zh', '.en'), fields = (chinese, english))

chinese.build_vocab(train_data, max_size=10000, min_freq=2) #can try 8000, 16000, or 32000
english.build_vocab(train_data, max_size=10000, min_freq=2)









def test():

    en_sent = 'testing the tokenizer!'
    zh_sent = "蘋果公司正考量用一億元買下英國的新創公司"
    doc = tokenize_zh(zh_sent)
    for word in doc:
        print(str(word) + " ")
    print("NOT tokenized :" + zh_sent)
    print(tokenize_en(en_sent))