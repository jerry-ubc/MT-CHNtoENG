from json import encoder
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k                         #Ger -> Eng dataset
from torchtext.data import Field, BucketIterator
import numpy as np                                              #loss plots
import spacy                                                    #tokenizer
import random
from torch.utils.tensorboard import SummaryWriter  #prints to tensorboard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ge = spacy.load('de')
spacy_en = spacy.load('en')

def tokenizer_ge(text):
    return [tok.text for tok in spacy_ge.tokenizer(text)]

def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

german = Field(tokenize=tokenizer_ge, lower = True,
               init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_en, lower = True,
               init_token='<sos>', eos_token='<eos>')

train_data, dev_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                  fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, p):
        #input_size: size of vocabulary (German)
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embed_size)               #(input) input_size -> embed_size (output)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=p)  #(input) embed_size -> input_size (output)

    def forward(self, x):
        #x: vector of indices to represent tokenized words/word components
        #   unpacking the information, you get a batch_size of sentences
        #x: (seq_length, batch_size)

        embedding = self.droppout(self.embedding(x))    #embedding: (seq_len, N, embed_size)
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers, p):
        #note input_size = output_size
        #input_size = size of English vocabulary
        #output is probabilities of each word in vocabulary
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=p)
        #hidden size of encoder and decoder are the same
        self.fc = nn.Linear(hidden_size, output_size)
        #fc means fully connected

    def forward(self, x, hidden, cell):
        #x: (batch_size), but we want (1, N) because decoder predicts 1 word at a time
        #given previous hidden and cell state, predict next work
        #recall encoder is (seq_len, N) since we send in entire German sentence as input
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))                     #embedding: (1, N, embed_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))   #outputs: (1, N, hidden_size)
        predictions = self.fc(outputs)                                  #predictions: (1, N, vocab_length)
        predictions = predictions.squeeze(0)                            #want (N, vocab_length) because ??????????????????????
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        #since model will sometimes predict incorrectly and use that incorrect prediction
        #as input for next word, fix it so it's not getting completely derailed
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        #we predict 1 word at-a-time for a batch
        #each prediction is a vector with length equal to entire vocabulary size
        #so 1 output will be sized: (batch_size, target_vocab_size)
        #add additional outputs on dim=0

        hidden, cell = self.encoder(source)
        x = target[0]   #x gets start token

        for word in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[word] = output

            #output: (N, english_vocab_size), argmax gives highest probability guess
            best_guess = output.argmax(1)
            x = target[word] if random.random() < teacher_force_ratio else best_guess


#training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

#model hyperparameters
load_model = False
device = torch.device('cua' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embed_size = 300
decoder_embed_size = 300
hidden_size = 1024 
num_layers = 4
enc_dropout = 0.5
dec_dropout = 0.5

#tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, dev_iterator, test_iterator = BucketIterator.splits((train_data, dev_data, test_data),
                                                                     batch_size = batch_size,
                                                                     sort_within_batch = True,
                                                                     sort_key = lambda x: len(x.src),
                                                                     device = device)   
                                                                     #all sentences vary in length, so BucketIterator prioritizes similar length examples to minimize number of padding
                                                                     #to save computations via sort_key parameter

encoder_net = Encoder(input_size_encoder, encoder_embed_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Encoder(input_size_decoder, decoder_embed_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)   #if receives a pad index, don't compute anything

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

sentence = ''

for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')

    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)
    model.eval()

    translated_sent = translate_sentence(model, sentence, german, english, device, max_length=50)

    print(f'Translated: \n {translated_sent}')

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        in_data = batch.src.to(device)
        target = batch.tgt.to(device)       #OR batch.trg.to(device)????????????????????????

        output = model(in_data, target) #output: (target_len, batch_size, output_dim)
                                        #currently output represents a target_len of words

        output = output[1:].output.reshape(-1, output.shape[2])
        #output[1:] removes output[0] as it's the start token
        #reshape turns output from 3D -> 2D, -1 means it's value is inferred

        target = target[1:].reshape(-1) #target: (target_len, batch_size) -> (target_len * batch_size)
        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)    #clips gradients to avoid exploding gradients
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1

score = bleu(test_data, model, german, english, device)
print(f'Bleu score {score*100:.2f}')