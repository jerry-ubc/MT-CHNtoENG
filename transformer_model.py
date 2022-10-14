from fnmatch import translate
import torch
import torch.optim as optim
import torch.nn as nn
import spacy    #vocabulary and tokenizer
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
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

german.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

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
src_vocab_size = len(german.vocab)
tgt_vocab_size = len(english.vocab)
embed_size = 512
num_heads = 8
num_encoder_layers = 3       #Attention is all you need uses 6?
num_decoder_layers = 3
dropout = 0.10              #seq2seq usually lower, can play around
max_len = 100               #max sentence length? also used for positional embedding
forward_expansion = 2048    #SHOULD BE 2048? MAYBE? comment says use default value, search it up
src_pad_idx = english.vocab.stoi["<pad>"]

#Tensorboard for plots
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, dev_iterator, test_iterator = BucketIterator.splits(
    (train_data, dev_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,               #SEARCH UP WHAT THIS DOES
    wort_key = lambda x: len(x.src)         #??????????????????????????????????????
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

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)       #ignores computation on padding to save resources

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

#sentence = ''

for epoch in range(num_epochs):
    print(f"[Epoch <epoch> / {num_epochs}")
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
    # model.eval()
    # translated_sentence = translate_sentence(
    #     model, sentence, german, english, device, max_length = 100
    # )
    # print(f"Example: \n {translated_sentence}")
    # model.train()

    for batch_index, batch in enumerate(train_iterator):
        in_data = batch.src.to(device)
        target = batch.trg.to(device)

        #forward prop
        output = model(in_data, target[:-1])    #ignore last element
        #want output to be 1 timestep ahead of input
        #e.g., input is <sos>, output is second element in target

        #suppose we have batch of 10 examples, then for every example in that batch, we have a predicted
        #translated sentence of, say 50 words, for each of those words we have another dimension which is length of target vocabulary
        output = output.reshape(-1, output.shape[2])        #FILL IN THIS DIMENSION FROM PREVIOUS EXAMPLES
        target = target[1:]     #want output to be 1 ahead of target (????) CONFIRM IT ISN'T OTHER WAY AROUND
        target = target.reshape(-1)
        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)      #prevents exploding gradient
        optimizer.step()
        writer.add_scalar("Training loss", loss, global_step = step)
        step += 1

score = bleu(test_data, model, german, english, device)
print(f"Bleu score {score*100:.2f}")