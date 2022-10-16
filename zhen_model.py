import torch
import torch.optim as optim
import torch.nn as nn
import spacy    #vocabulary and tokenizer
import spacy
from dataset import NCDataset
from transformer_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter   #loss plots
from torchtext.datasets import Multi30k, TranslationDataset
from torchtext.data import Field, BucketIterator
from spacy.lang.zh import Chinese
#from zhutils import NCDataset

# spacy_zh = spacy.load('zh_core_web_trf')        #MAYBE CAN COMMENT OUT!!!!!
# spacy_en = spacy.load('en_core_web_trf')

spacy_zh = spacy.load('zh_core_web_sm')        #MAYBE CAN COMMENT OUT!!!!!
spacy_en = spacy.load('en_core_web_sm')


#------------------------use these later
# Jieba
# cfg = {"segmenter": "jieba"}
# nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})

# #tokenizers
# def tokenize_zh(text):
#     return nlp(text)
#-----------------------------------------------
def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


#tokenizer parameters
chinese = Field(tokenize=tokenize_zh, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenize_en, lower=True, init_token='<sos>', eos_token='<eos>')

#RUN corpus_process.py FIRST!
train_data = NCDataset(src_data_path='.data/ncv16/train.zh', tgt_data_path='.data/ncv16/train.en')
dev_data = NCDataset(src_data_path='.data/ncv16/val.zh', tgt_data_path='.data/ncv16/val.en')
test_data = NCDataset(src_data_path='.data/ncv16/test.zh', tgt_data_path='.data/ncv16/test.en')

class Transformer(nn.Module):
    def __init__(self, 
                 embed_size, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 src_pad_idx, 
                 num_heads, 
                 num_encoder_layers,
                 num_decoder_layers,
                 forward_expansion,
                 dropout,
                 max_len,
                 device
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_len, embed_size)
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.tgt_position_embedding = nn.Embedding(max_len, embed_size)
        self.device = device
        self.transformer = nn.Transformer(embed_size,
                                          num_heads,
                                          num_encoder_layers,
                                          num_decoder_layers,
                                          forward_expansion,
                                          dropout
        )
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        #src: (src_len, N), but PyTorch's transformer takes input (N, src_len)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask

    def forward(self, src, tgt):
        src_seq_len, N = src.shape
        tgt_seq_len, N = tgt.shape

        src_positions = (
            torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device)
        )
        tgt_positions = (
            torch.arange(0, tgt_seq_len).unsqueeze(1).expand(tgt_seq_len, N).to(self.device)
        )
        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_tgt = self.dropout(
            (self.tgt_word_embedding(tgt) + self.tgt_position_embedding(tgt_positions))
        )
        src_padding_mask = self.make_src_mask(src)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

        out = self.transformer(
            embed_src,
            embed_tgt,
            embed_key_padding_mask = src_padding_mask,
            tgt_mask = tgt_mask
        )
        out = self.fc_out(out)                  # WHAT DOES THIS DO??????????/
        return out

#Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

#Training hyperparameters
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

#Model hyperparameters
src_vocab_size = 32000          #THESE ARE MAGIC NUMBERS, PARAMETRIZE THEM LATER
tgt_vocab_size = 32000
embed_size = 512
num_heads = 8
num_encoder_layers = 3       #Attention is all you need uses 6?
num_decoder_layers = 3
dropout = 0.10              #seq2seq usually lower, can play around
max_len = 100               #max sentence length? also used for positional embedding
forward_expansion = 2048    #SHOULD BE 2048? MAYBE? comment says use default value, search it up
# src_pad_idx = english.vocab.stoi["<pad>"]
src_pad_idx = train_data.spc_en.pad_id()

writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, dev_iterator, test_iterator = BucketIterator.splits(
    (train_data, dev_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,               #SEARCH UP WHAT THIS DOES
    sort_key = lambda x: len(x.src),        #??????????????????????????????????????
    device = device
)

model = Transformer(
    embed_size,
    src_vocab_size,
    tgt_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device
).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)      #SEARCH UP OTHER OPTIMIZERS???????????

pad_idx = train_data.spc_en.pad_id()
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)       #ignores computation on padding to save resources

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)











def test():

    en_sent = 'testing the tokenizer!'
    zh_sent = "蘋果公司正考量用一億元買下英國的新創公司"
    doc = tokenize_zh(zh_sent)
    for word in doc:
        print(str(word) + " ")
    print("NOT tokenized :" + zh_sent)
    print(tokenize_en(en_sent))